from fastapi import FastAPI
from typing import List, Dict, Any
from datetime import datetime, timedelta
import uuid

app = FastAPI(title="Mock MCP Feedback Server")

# Pre-generate a small rotating buffer of items
BASE_TIME = datetime.now()


def _item(idx: int) -> Dict[str, Any]:
	return {
		"trace_id": f"mock-{idx}-{uuid.uuid4().hex[:8]}",
		"prompt": "Translate to Hindi: Hello, how are you?",
		"response": "नमस्ते, आप कैसे हैं?",
		"correction": None,
		"reward": 1.0 if idx % 2 == 0 else 0.5,
		"timestamp": (BASE_TIME + timedelta(seconds=idx)).isoformat()
	}

MOCK_ITEMS: List[Dict[str, Any]] = [_item(i) for i in range(1, 21)]


@app.get("/feedback")
async def get_feedback(limit: int = 50, since: str | None = None):
	items = MOCK_ITEMS
	if since:
		try:
			items = [it for it in items if it.get("timestamp", "") > since]
		except Exception:
			pass
	return items[:limit]
