# Sovereign Core: KSML + MCP + RL + Vaani (Concise README)

## Overview
- Live pipeline: LM Core → KSML aligner → (optional) Vaani → logs.
- Continuous improvement: MCP stream pulls feedback → RL policy updates → KSML nudges.

## Key Components
- KSML aligner: `sovereign_core/ksml/aligner.py`
  - Classifies intent, karma, source/target languages, Sanskrit roots.
  - Applies RL nudges: tone, confidence delta, optional target_lang override.
- MCP stream: `sovereign_core/mcp/stream_client.py`
  - Polls connectors (HTTP), dedupes by `trace_id`, appends `data/feedback_stream.jsonl`.
  - Health/telemetry to `logs/ksml_bridge.jsonl`.
- RL policy: `sovereign_core/rl/policy.py`, `sovereign_core/rl/policy_updater.py`
  - Updates a simple bandit table from feedback; writes `data/rl_policy.json` and `data/rl_sync.json`.
  - Periodic sync (S3/NAS placeholder).
- Bridge (orchestration): `sovereign_core/bridge/reasoner.py`
  - Calls LM Core `/compose.final_text`, runs KSML, logs, optional Vaani.

## Run (local)
- Enable MCP stream with API:
  - CMD:
    - `set MCP_STREAM_ENABLED=1`
    - `set PYTHONPATH=.`
    - `.\venv\Scripts\python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8116`
- Optional mock feedback server:
  - `.\venv\Scripts\python -m uvicorn sovereign_core.mcp.mock_feedback_server:app --host 127.0.0.1 --port 8120`
- Verify files:
  - `data/feedback_stream.jsonl` (feedback)
  - `logs/ksml_bridge.jsonl` (bridge/health)

## RL + KSML quick smoke
- `set PYTHONPATH=.` then:
  - `.\venv\Scripts\python scripts\smoke_rl_ksml.py`
- Expect KSML output with tone/confidence nudges; target_lang override triggers if recent feedback is translation-heavy.

## Config (.env)
- MCP: set `MCP_STREAM_ENABLED=1` to start background poller.
- Vaani: set `VAANI_URL`, `VAANI_USERNAME`, `VAANI_PASSWORD` to enable TTS.
- LM Core endpoint is configured in `sovereign_core/bridge/reasoner.py`.

## S3/NAS sync (for RL snapshots)
- File produced: `data/rl_sync.json` (policy + last rewards).
- What infra provides:
  - S3: `RL_S3_BUCKET`, `RL_S3_PREFIX`, `AWS_DEFAULT_REGION` (+ IAM role or keys).
  - NAS: `RL_NAS_PATH` (SMB/NFS), credentials or machine access.
- Where to implement upload: `sovereign_core/rl/policy_updater.py::_sync_to_s3` (TODO marked).
- Add `boto3` to `requirements.txt` if using S3.

## Logs and Targets
- Latency target: ≤ 2s end-to-end; VRAM ≤ 4GB (RTX 4050).
- Accuracy target: KSML tags ≥ 85% consistency.
- All major events logged to `logs/ksml_bridge.jsonl`.
