# LM_KSML_MCP_RL_HANDOVER.md — Sovereign LM, KSML, MCP & RL Handover

**Subsystem**: Sovereign LM, Multilingual Tokenization, Model Context Protocol (MCP), Reinforcement Learning (RL)  
**Receiving Owners**: Vijay Dhawan + Harsha  
**Previous Owner**: Soham Kotkar  
**Primary Repository**: [Multilingual-Tokenization-Model-Integration](file:///c:/PC/Office%20Projects/Multilingual-Tokenization-Model-Integration) (`https://github.com/Soham20030/Multilingual.git`)  
**Active Branch**: `main`  
**Last Verified Date**: 13 August 2026  
**Status**: **PROVISIONAL / WORKING** (Checkpoints & Inference Tested)  

---

## 1. System Overview & Architecture

This repository encapsulates the multilingual tokenization and Sovereign LM model adapters, allowing standard LLM requests to interface with Model Context Protocol (MCP) servers, knowledge bases, and KSML structured representation layers.

```text
[Incoming Multilingual User Query (21 Languages supported)]
                      │
                      ▼
       [FastAPI Application: main.py / src/api/main.py]
                      ├── /qa                       ──► [Knowledge Base RAG & Prompt Engineering]
                      ├── /multilingual-conversation ──► [Session Context & Multi-turn History]
                      ├── /test-language-switching   ──► [Mid-Conversation Language Switcher]
                      ├── /tokenize                 ──► [SentencePiece & HF Tokenizer Pipeline]
                      └── Model Inference Core      ──► [Unsloth Llama-3.2-3B Instruct 4-bit]
                                                             ├── Checkpoint: checkpoint-125000 (Epoch 5.0)
                                                             ├── Primary: CUDA GPU Inference
                                                             └── Fallback: Automatic CPU Mode Switcher
```

---

## 2. Key Components & Implementation

1. **Model Checkpoint Lineage**:
   - Checkpoint path: `checkpoints/checkpoint-125000/checkpoint_info.pkl`
   - Base Model: `unsloth/llama-3.2-3b-instruct-bnb-4bit` (Step 125,000, Epoch 5.0)
2. **Robust CUDA & CPU Fallback Engine**:
   - `reload_model_cpu_only()` automatically catches Windows PyTorch DLL (`WinError 1114`) and CUDA assertion errors, dynamically switching execution to CPU without crashing the server.
3. **Model Context Protocol (MCP) Stream Client**:
   - Interoperates with `sovereign_core.mcp.stream_client.MCPStreamClient` for agent tool execution.
4. **Mid-Conversation Language Switching**:
   - Endpoint `/test-language-switching` tests on-the-fly switching across Hindi, Sanskrit, Marathi, Tamil, Telugu, Bengali, and English.

---

## 3. Verification & Execution Instructions

### Local Execution (Python / Uvicorn)
```bash
cd "c:\PC\Office Projects\Multilingual-Tokenization-Model-Integration"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Docker Execution
```bash
cd "c:\PC\Office Projects\Multilingual-Tokenization-Model-Integration"
docker build -t bhiv/multilingual-lm:latest .
docker run -p 8080:8080 bhiv/multilingual-lm:latest
```

### Verification Endpoints
- **Healthcheck**: `GET http://localhost:8080/health`
- **Q&A KB API**: `POST http://localhost:8080/qa`
- **Language Switch Test**: `POST http://localhost:8080/test-language-switching`
- **Inference Verification Script**: `python test_checkpoint_125000_simple.py`

---

## 4. Known Issues & Operational Recommendations

1. **PyTorch CUDA DLL Error on Windows**: If PyTorch throws `WinError 1114`, use WSL (`venv_wsl/bin/activate`) or set environment variable `FORCE_CPU_GENERATION=true`.
2. **Model Download Cold Start**: Initial container startup will download base HuggingFace weights (~4GB). Bake weights into persistent volume for production deployments.

---

## 5. Ownership Transfer Sign-off

- **Previous Owner**: Soham Kotkar
- **Receiving Owners**: Vijay Dhawan + Harsha
- **Transfer Status**: **READY** (Models, API routes, CPU fallback verified)
