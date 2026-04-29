"""
MCP Feedback Stream Client - KSML / RL Feedback Ingestion

High-level role (to disambiguate from top-level mcp/stream_client.py):
- This file streams **feedback items** (trace_id, prompt, response, correction, reward)
- It pulls from one or more HTTP feedback APIs using `sovereign_core/mcp/config.yml`
- Output is appended to **sovereign_core/data/feedback_stream.jsonl** and logged to
  `sovereign_core/logs/ksml_bridge.jsonl` for KSML + RL.

It does NOT stream raw corpus text for training; that is handled by `mcp/stream_client.py`.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
	import httpx
except ImportError:  # pragma: no cover
	httpx = None  # type: ignore

try:
	import yaml
except ImportError:  # pragma: no cover
	yaml = None  # type: ignore

_BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(_BASE_DIR, "config.yml")
WATERMARKS_PATH = os.path.join(_BASE_DIR, ".watermarks.json")
SEEN_IDS_PATH = os.path.join(_BASE_DIR, ".seen_ids.json")
HEALTH_LOG_PATH = os.path.join("logs", "ksml_bridge.jsonl")


class MCPStreamClient:
	def __init__(self, config_path: str = CONFIG_PATH) -> None:
		self.config_path = config_path
		self.config = self._load_config(config_path)
		self.poll_interval = int(self.config.get("poll_interval_seconds", 5))
		self.max_backoff = int(self.config.get("max_backoff_seconds", 60))
		self.request_timeout = int(self.config.get("request_timeout_seconds", 10))
		self.health_logging = bool(self.config.get("health_logging", True))
		self.sink_path = self.config.get("sink", {}).get("path", "data/feedback_stream.jsonl")
		self.sink_ensure_ascii = bool(self.config.get("sink", {}).get("ensure_ascii", False))
		self.connectors = list(self.config.get("connectors", []))

		os.makedirs(os.path.dirname(self.sink_path), exist_ok=True)
		os.makedirs(os.path.dirname(HEALTH_LOG_PATH), exist_ok=True)
		self._watermarks = self._load_json(WATERMARKS_PATH, default={})
		self._seen_ids = set(self._load_json(SEEN_IDS_PATH, default=[]))

	def _load_config(self, path: str) -> Dict[str, Any]:
		if yaml is None:
			raise RuntimeError("pyyaml is required to load MCP config")
		with open(path, "r", encoding="utf-8") as f:
			return yaml.safe_load(f) or {}

	def _load_json(self, path: str, default: Any) -> Any:
		try:
			with open(path, "r", encoding="utf-8") as f:
				return json.load(f)
		except FileNotFoundError:
			return default
		except Exception:
			return default

	def _save_json(self, path: str, data: Any) -> None:
		try:
			with open(path, "w", encoding="utf-8") as f:
				json.dump(data, f, ensure_ascii=False, indent=2)
		except Exception:
			pass

	def _log_health(self, entry: Dict[str, Any]) -> None:
		if not self.health_logging:
			return
		entry_with_time = {**entry, "timestamp": datetime.now().isoformat()}
		try:
			with open(HEALTH_LOG_PATH, "a", encoding="utf-8") as f:
				f.write(json.dumps(entry_with_time, ensure_ascii=False) + "\n")
		except Exception:
			pass

	def _get_auth_header(self, connector: Dict[str, Any]) -> Dict[str, str]:
		token_env_key = connector.get("auth_env_token")
		if not token_env_key:
			return {}
		token_value = os.getenv(token_env_key)
		if token_value:
			return {"Authorization": f"Bearer {token_value}"}
		return {}

	def _build_url_and_params(self, connector: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
		url = connector.get("url")
		params = dict(connector.get("params", {}))
		wm_key = f"watermark::{connector.get('name','unknown')}"
		last_since = self._watermarks.get(wm_key)
		if last_since is not None:
			params["since"] = last_since
		return url, params

	async def _fetch_connector_http(self, client: httpx.AsyncClient, connector: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
		name = connector.get("name", "unknown")
		method = (connector.get("method") or "GET").upper()
		headers = dict(connector.get("headers", {}))
		headers.update(self._get_auth_header(connector))
		url, params = self._build_url_and_params(connector)
		start = time.time()
		try:
			resp = await client.request(method, url, params=params, headers=headers, timeout=self.request_timeout)
			resp.raise_for_status()
			data = resp.json()
			if isinstance(data, dict) and "items" in data:
				items = data.get("items", [])
			else:
				items = data if isinstance(data, list) else []
			latency = time.time() - start
			return name, [i for i in items if isinstance(i, dict)], None
		except Exception as e:
			latency = time.time() - start
			self._log_health({
				"component": "mcp.stream_client",
				"event": "connector_error",
				"connector": name,
				"error": str(e),
				"latency_ms": int(latency * 1000),
			})
			return name, [], str(e)

	def _validate_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		required = ["trace_id", "prompt", "response", "timestamp"]
		for k in required:
			if k not in item:
				return None
		if "correction" not in item:
			item["correction"] = None
		if "reward" not in item:
			item["reward"] = 0.0
		return item

	def _dedupe(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		new_items: List[Dict[str, Any]] = []
		for it in items:
			trace_id = str(it.get("trace_id"))
			if trace_id and trace_id not in self._seen_ids:
				self._seen_ids.add(trace_id)
				new_items.append(it)
		return new_items

	def _append_sink(self, items: List[Dict[str, Any]]) -> int:
		if not items:
			return 0
		count = 0
		with open(self.sink_path, "a", encoding="utf-8") as f:
			for it in items:
				f.write(json.dumps(it, ensure_ascii=self.sink_ensure_ascii) + "\n")
				count += 1
		return count

	def _advance_watermark(self, connector_name: str, items: List[Dict[str, Any]]) -> None:
		if not items:
			return
		wm_key = f"watermark::{connector_name}"
		try:
			max_ts = max(it.get("timestamp", "") for it in items)
			if max_ts:
				self._watermarks[wm_key] = max_ts
				self._save_json(WATERMARKS_PATH, self._watermarks)
		except Exception:
			pass

	async def poll_once(self) -> Dict[str, Any]:
		if httpx is None:
			raise RuntimeError("httpx is required for MCP streaming")
		stats = {"total_received": 0, "total_written": 0, "per_connector": {}}
		async with httpx.AsyncClient() as client:
			for conn in self.connectors:
				name = conn.get("name", "unknown")
				if (conn.get("type") or "http").lower() != "http":
					continue
				cname, items, err = await self._fetch_connector_http(client, conn)
				valid = [self._validate_item(i) for i in items]
				valid = [i for i in valid if i is not None]
				new_items = self._dedupe(valid)
				written = self._append_sink(new_items)
				self._advance_watermark(name, valid)

				stats["total_received"] += len(items)
				stats["total_written"] += written
				stats["per_connector"][name] = {
					"received": len(items),
					"valid": len(valid),
					"new": len(new_items),
					"written": written,
					"error": err,
				}
		self._save_json(SEEN_IDS_PATH, list(self._seen_ids))
		self._log_health({"component": "mcp.stream_client", "event": "poll_once", **stats})
		return stats

	async def run_forever(self) -> None:
		backoff = self.poll_interval
		while True:
			try:
				stats = await self.poll_once()
				backoff = self.poll_interval if stats.get("total_received", 0) > 0 else min(max(backoff * 2, self.poll_interval), self.max_backoff)
			except Exception as e:
				self._log_health({"component": "mcp.stream_client", "event": "run_error", "error": str(e)})
				backoff = min(max(backoff * 2, self.poll_interval), self.max_backoff)
			await asyncio.sleep(backoff)


async def _main() -> None:
	client = MCPStreamClient(CONFIG_PATH)
	await client.run_forever()


if __name__ == "__main__":
	if not os.path.exists(CONFIG_PATH):
		print(f"Config not found: {CONFIG_PATH}", file=sys.stderr)
		sys.exit(1)
	try:
		asyncio.run(_main())
	except KeyboardInterrupt:
		print("MCP stream stopped.")
