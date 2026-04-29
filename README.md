# BHIV Sovereign AI Platform 🚀

**An integrated multilingual AI platform combining:**
1. **Sovereign LM Bridge** - Multilingual reasoning with KSML alignment, RL, and MCP streaming
2. **BHIV Core** - Multi-modal AI processing with reinforcement learning and Named Learning Objects

---

## 🌟 Platform Overview

This repository houses two integrated AI systems that work together:

### System 1: Sovereign LM Bridge + Multilingual KSML Core
**Production-ready multilingual generation system** with KSML (Knowledge, Semantic, Multilingual, Language) semantic alignment, RL-based self-improvement, and MCP streaming.

**Key Features:**
- ✅ Multilingual generation API (31 languages: 21 Indian + 10 South Asian)
- ✅ KSML Semantic Alignment Engine (intent, karma state, Sanskrit roots)
- ✅ RL Self-Improvement Loop with policy updates
- ✅ MCP-Driven Feedback Stream (HF + S3 + HTTP + Qdrant connectors)
- ✅ Vaani TTS Compatibility Layer (prosody-optimized speech-ready output)
- ✅ Request queuing + model caching (production-ready)
- ✅ Memory management optimized for RTX 4050

### System 2: BHIV Core - Multi-Modal AI Processing
**Advanced AI processing pipeline** with multi-modal input support, reinforcement learning, Named Learning Object (NLO) generation, and production-ready web interface.

**Key Features:**
- ✅ Multi-Modal Processing (text, PDF, image, audio inputs)
- ✅ Reinforcement Learning (UCB-based agent selection)
- ✅ Named Learning Objects with Bloom's taxonomy
- ✅ Web Interface (Bootstrap UI with authentication)
- ✅ MongoDB Integration (persistent NLO storage)
- ✅ Enhanced CLI (batch processing with progress bars)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- MongoDB 5.0+ (for BHIV Core NLO storage)
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU with CUDA support (RTX 4050 or better)
- API Keys: Groq, Qdrant, MongoDB Atlas (configure in `.env`)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sohamkotkar2003/Multilingual-Tokenization-Model-Integration.git
cd Multilingual-Tokenization-Model-Integration

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install NLP models (for BHIV Core)
python -m spacy download en_core_web_sm

# 5. Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Running the Systems

#### Sovereign LM Bridge (Port 8116)
```bash
# Start the Sovereign KSML API
cd C:\pc\Project
venv\Scripts\activate
set PYTHONPATH=.
set MCP_STREAM_ENABLED=1
python -m uvicorn sovereign_core.api:app --host 127.0.0.1 --port 8116
```

**Endpoints:**
- `POST /align.ksml` - KSML semantic alignment
- `POST /rl.feedback` - RL policy updates
- `POST /compose.speech_ready` - Vaani TTS-ready output
- `POST /bridge.reason` - Unified multilingual reasoning pipeline
- `GET /health` - Health check

#### LM Core Adapter (Port 8117)
```bash
# Start the LM Core API (Bhavesh's integration)
cd C:\pc\Project
venv\Scripts\activate
set PYTHONPATH=.
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8117
```

**Endpoints:**
- `POST /compose.final_text` - Multilingual text generation with RAG
- `POST /compose.stream` - Streaming generation
- `GET /health` - Health check

#### BHIV Core MCP Bridge (Port 8002)
```bash
# Start MongoDB (ensure it's running first)
# sudo systemctl start mongod  # Linux
# net start MongoDB  # Windows

# Start the BHIV MCP Bridge
python mcp_bridge.py
```

#### BHIV Web Interface (Port 8003)
```bash
# Start the Web UI (in another terminal)
python integration/web_interface.py
```

**Access:**
- Web Interface: http://localhost:8003 (admin/secret or user/secret)
- API Documentation: http://localhost:8002/docs
- Health Checks: http://localhost:8002/health

---

## 📂 Project Structure

```
├── sovereign_core/              # Sovereign LM Bridge + KSML
│   ├── api.py                  # FastAPI endpoints (align, rl, speech, bridge)
│   ├── ksml/                   # KSML semantic alignment engine
│   │   ├── aligner.py          # Intent + karma + Sanskrit root tagging
│   │   └── ksml_roots.json     # Sanskrit roots + meanings
│   ├── rl/                     # Reinforcement Learning
│   │   ├── policy.py           # Policy manager (Q-table/bandit)
│   │   └── policy_updater.py   # RL self-improvement loop
│   ├── mcp/                    # MCP feedback stream
│   │   ├── stream_client.py    # Polling, dedup, watermarking
│   │   ├── config.yml          # Connector endpoints
│   │   └── mock_feedback_server.py  # Test server
│   ├── bridge/
│   │   ├── vaani_adapter.py    # Vaani TTS compatibility
│   │   └── speech_composer.py  # Prosody-optimized output
│   └── README.md               # Sovereign Core docs
│
├── lm_core_adapter/            # LM Core integration (Bhavesh)
│   └── app.py                  # FastAPI server for /compose.final_text
│
├── src/                        # Shared services
│   ├── api/
│   │   └── main.py             # Main API endpoint
│   ├── services/
│   │   ├── kb_service.py       # Knowledge base (Qdrant)
│   │   └── knowledge_base.py   # RAG integration
│   └── integration/
│       └── lm_connector.py     # LM Core connector
│
├── agents/                     # BHIV Core Agents
│   ├── text_agent.py           # Text processing
│   ├── archive_agent.py        # PDF processing
│   ├── image_agent.py          # Image processing (BLIP)
│   ├── audio_agent.py          # Audio processing (Wav2Vec2)
│   ├── agent_registry.py       # Dynamic agent configuration
│   └── stream_transformer_agent.py
│
├── integration/                # BHIV Core Integration
│   ├── llm_router.py           # LLM routing logic
│   ├── nipun_adapter.py        # NLO generator
│   └── web_interface.py        # Web UI (FastAPI + Jinja2)
│
├── reinforcement/              # BHIV RL System
│   ├── agent_selector.py       # UCB-based agent selection
│   ├── model_selector.py       # RL model selection
│   ├── replay_buffer.py        # Experience replay
│   ├── reward_functions.py     # Reward calculation
│   └── retrain_rl.py           # Automated retraining
│
├── mcp_bridge.py               # BHIV MCP Bridge API
├── simple_api.py               # BHIV Simple API
├── cli_runner.py               # Enhanced CLI
│
├── scripts/                    # Testing & utilities
│   ├── send_rl_feedback.py     # Send RL feedback
│   ├── smoke_rl_ksml.py        # Smoke test RL + KSML
│   └── test_endpoints.py       # Test all endpoints
│
├── data/                       # Training data
│   ├── training/               # 21 language training files
│   ├── validation/             # Validation data
│   └── feedback_stream.jsonl   # MCP feedback stream
│
├── logs/                       # Logging
│   ├── ksml_bridge.jsonl       # KSML bridge logs
│   ├── agent_logs.json         # BHIV agent logs
│   ├── learning_log.json       # BHIV RL logs
│   └── model_logs.json         # BHIV model logs
│
├── docs/                       # Documentation
│   ├── BHIV_Core_MCP_Connector_Documentation.md
│   ├── complete_usage_guide.md
│   ├── mcp_api.md
│   ├── reinforcement.md
│   └── nlo_schema.md
│
├── templates/                  # BHIV Web UI templates
│   ├── base.html
│   ├── index.html
│   └── dashboard.html
│
├── .env                        # Environment variables (API keys)
├── requirements.txt            # All dependencies
└── README.md                   # This file
```

---

## 🌍 Supported Languages (Sovereign LM Bridge)

### **21 Indian Languages**

- Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Urdu, Punjabi
- Kannada, Malayalam, Assamese, Odia, Sanskrit, Nepali, Sindhi
- Kashmiri, Maithili, Bodo, Meitei, Santali, English

Base model: `bigscience/bloomz-560m` + Gurukul Lite adapter

---

## 📡 MCP Streaming (Sovereign)

**No Large Downloads** - Stream data from:
- **HuggingFace datasets** - 50,000+ datasets
- **S3/Cloud storage** - AWS S3, Azure Blob, GCS
- **HTTP APIs** - RESTful streaming endpoints
- **Qdrant Vector DB** - Vector database integration
- **Automatic fallback** - Falls back to local data
- **Memory efficient** - <100MB data buffer

---

## 🎮 Reinforcement Learning

### Sovereign RL (Policy-based)
- **Policy Updates**: Q-table/bandit-style reward learning
- **RL Feedback**: `/rl.feedback` endpoint accepts `{prompt, output, reward}`
- **NAS/S3 Sync**: Automatic policy snapshot uploads
- **Alignment Influence**: Policy nudges KSML alignment (karma, tone, confidence)

### BHIV RL (UCB-based)
- **Dynamic Agent Selection**: UCB optimization
- **Task-complexity-based exploration**
- **Automatic retraining** from historical data
- **Replay buffer** for experience storage

---

## 📊 Performance

| Metric | Sovereign LM Bridge | BHIV Core |
|--------|---------------------|-----------|
| **Languages** | 21+ | Multi-lingual support |
| **Latency (end-to-end)** | ≤ 2s | Varies by agent |
| **GPU Memory** | ≤ 4 GB (RTX 4050) | ~300MB |
| **MCP throughput** | ~1.1 samples/sec | N/A |
| **API response** | <1s (excluding generation) | <1s |

---

## 📚 Documentation

### Sovereign LM Bridge
- **[Sovereign Core README](sovereign_core/README.md)** - Detailed KSML + RL + MCP docs
- **[Smoke Test Results](results/smoke_results_20251029.md)** - Test results

### BHIV Core
- **[BHIV System Summary](BHIV_System_Summary_for_Vinayak.md)** - System overview
- **[MCP Connector Docs](BHIV_Core_MCP_Connector_Documentation.md)** - MCP integration
- **[Complete Usage Guide](docs/complete_usage_guide.md)** - Full guide
- **[NLO Schema](docs/nlo_schema.md)** - Named Learning Objects
- **[Reinforcement Learning](docs/reinforcement.md)** - RL details
- **[API Documentation](docs/mcp_api.md)** - API reference

---

## 🔗 System Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                    BHIV Sovereign AI Platform                   │
└─────────────────────────────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│  Sovereign LM      │  │   LM Core Adapter  │  │   BHIV Core MCP    │
│  Bridge (8116)     │  │      (8117)        │  │   Bridge (8002)    │
│  - KSML Alignment  │  │  - RAG (Qdrant)    │  │  - Multi-Modal     │
│  - RL Policy       │  │  - MongoDB Logs    │  │  - NLO Generation  │
│  - MCP Stream      │  │  - Streaming Gen   │  │  - Agent Registry  │
│  - Vaani TTS       │  │                    │  │  - RL Selection    │
└────────────────────┘  └────────────────────┘  └────────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Unified Frontend  │
                    │   (Web UI: 8003)    │
                    └─────────────────────┘
```

---

## ⚙️ Configuration

Key environment variables (`.env`):

```bash
# Groq API (LM Core)
GROQ_API_KEY_MAIN=your_groq_key
GROQ_API_KEY_FALLBACK=your_fallback_key

# MongoDB
MONGODB_URI=your_mongodb_uri

# Qdrant (RAG)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=documents

# Vaani TTS (optional)
VAANI_URL=http://localhost:8080
VAANI_USERNAME=user
VAANI_PASSWORD=pass
VAANI_VOICE=default
VAANI_DEFAULT_LANG=en

# Sovereign API
SOVEREIGN_HOST=127.0.0.1
SOVEREIGN_PORT=8116

# MCP Stream
MCP_STREAM_ENABLED=1

# RL NAS/S3 Sync
RL_NAS_PATH=\\192.168.0.90\Soham_Kotkar
# RL_S3_BUCKET=bhiv
# RL_S3_PREFIX=rl_feedback/sovereign_core/
```

---

## 🧪 Testing

```bash
# Activate environment
venv\Scripts\activate
set PYTHONPATH=.

# Test Sovereign LM Bridge
python scripts/test_endpoints.py

# Test RL + KSML integration
set PYTHONIOENCODING=utf-8
python scripts/smoke_rl_ksml.py

# Send RL feedback (10 samples)
python scripts/send_rl_feedback.py http://127.0.0.1:8116/rl.feedback

# Test BHIV Core
python mcp_test.py
python test_audio_agent.py
python test_image_agent.py
```

---

## 👥 Team & Coordination

| Area | Collaborator | Responsibility |
|------|--------------|----------------|
| LM Response Source | Bhavesh | LM Core API endpoint |
| TTS Mapping | Karthikeya | Vaani tone & prosody schema |
| Feedback & RL Storage | Vijay | S3/NAS endpoint for reward uploads |
| MCP Streams | Nipun | Core dataset connectors |
| Testing & Task Bank | Vinayak | Pipeline validation |

---

## 🐛 Known Issues

### Sovereign LM Bridge
- ✅ All systems operational
- ⚠️ NAS sync requires valid network path (`RL_NAS_PATH`)
- ⚠️ MCP connectors require configuration in `sovereign_core/mcp/config.yml`

### BHIV Core
- ⚠️ MongoDB must be running for NLO storage
- ⚠️ Image/audio agents require model downloads on first run

---

## 📝 License

This project is part of the BHIV initiative for sovereign AI development.

---

## 🙏 Acknowledgments

Built with:
- 🤗 HuggingFace Transformers
- ⚡ FastAPI
- 🧠 PyTorch
- 🗄️ MongoDB, Qdrant
- 🎨 Bootstrap (Web UI)

---

**For detailed component documentation, see the respective README files in each module.**
