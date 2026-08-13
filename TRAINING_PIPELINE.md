# Multilingual Tokenization & Fine-Tuning Training Pipeline 🚀

This document provides a detailed technical breakdown of the entire model training, fine-tuning, data preprocessing, checkpointing, and reinforcement learning (RL) pipeline implemented in this project.

---

## 📐 Pipeline Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA PREPROCESSING & CLEANING                                                             │
│  Raw Corpora (sovereign_data/training/) ──> Indic Cleaners (sovereign_data_cleaning/*.py)   │
│  ──> Formatted Instruction JSONL (data/Hindi.jsonl, Marathi.jsonl, Bengali.jsonl, etc.)    │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. MODEL FINE-TUNING (UNSLOTH + QLoRA)                                                     │
│  Base Model: unsloth/Llama-3.2-3B-Instruct-bnb-4bit (or BLOOMZ-560M)                        │
│  Script: unsloth/unsloth_train_wsl.py                                                       │
│  Trainer: SFTTrainer (TRL) with 4-bit Quantization & LoRA Adapters                          │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. CHECKPOINT MANAGEMENT                                                                    │
│  Output Directory: checkpoints/ (e.g., checkpoint-120000, checkpoint-125000)                │
│  Artifacts: adapter_model.safetensors, adapter_config.json, checkpoint_info.pkl             │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. INFERENCE & EVALUATION                                                                   │
│  Scripts: test_checkpoint_125000_inference.py, test_model_simple.py                         │
│  Pipeline: Base Model + LoRA Adapter Loading ──> Token Generation & Multilingual Checks      │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. REINFORCEMENT LEARNING & KSML ALIGNMENT                                                  │
│  KSML Aligner: sovereign_core/ksml/aligner.py (Intent, Karma, Sanskrit Roots)              │
│  RL Feedback Loop: reinforcement/retrain_rl.py & sovereign_core/rl/                         │
│  Policy Update: UCB-based agent selection & policy updates via /rl.feedback                │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Phase 1: Data Ingestion & Script-Specific Cleaning

### Raw Datasets (`sovereign_data/training/`)
The raw training data consists of monolingual and parallel corpora across **21 Indian languages**, South Asian languages, Arabic, and English. 
Files in `sovereign_data/training/` include:
- Indic scripts: `hi_train.txt` (Hindi), `mr_train.txt` (Marathi), `bn_train.txt` (Bengali), `ta_train.txt` (Tamil), `te_train.txt` (Telugu), `gu_train.txt` (Gujarati), `kn_train.txt` (Kannada), `ml_train.txt` (Malayalam), `pa_train.txt` (Punjabi), `or_train.txt` (Odia), `sa_train.txt` (Sanskrit), `mai_train.txt` (Maithili), `ne_train.txt` (Nepali), `sd_train.txt` (Sindhi), `ks_train.txt` (Kashmiri), `as_train.txt` (Assamese), `sat_train.txt` (Santali), `mni_train.txt` (Meitei), `bo_train.txt` (Bodo).
- Additional: `en_train.txt` / `en_train.csv` (English), `ur_train.txt` (Urdu).

### Language-Specific Cleaning Modules (`sovereign_data_cleaning/`)
To ensure high-quality tokenization and prevent script corruption during tokenization, custom cleaning scripts are provided in `sovereign_data_cleaning/`:

- **Key Data Cleaning Tasks:**
  1. **Unicode Normalization:** Normalizing Devanagari, Dravidian, Bengali, and Perso-Arabic unicode points (NFC/NFD normalization).
  2. **Noise & Garbage Filtering:** Stripping invalid characters, broken URL strings, repetitive punctuation, and corrupted HTML/XML tags.
  3. **Script Isolation:** Ensuring target language sentences contain appropriate script ranges (e.g., Devanagari range `U+0900` to `U+097F`, Bengali `U+0980` to `U+09FF`, etc.).
  4. **Token Boundary Preservation:** Preserving matra/halant combinations so subword tokenizers do not split dependent vowels from consonants incorrectly.

---

## 📋 Phase 2: Instruction Dataset Preparation (`data/`)

Cleaned text data is assembled into structured JSONL instruction files located in `data/`:
- `Hindi.jsonl` (~652 MB)
- `Marathi.jsonl` (~697 MB)
- `Bengali.jsonl` (~294 MB)
- `Arabic.jsonl` (~809 MB)
- `English.jsonl` (~374 MB)

### JSONL Schema Structure
Each record follows an instruction-response format:
```json
{
  "instruction": "Explain the importance of artificial intelligence in education in Marathi.",
  "input": "",
  "output": "कृत्रिम बुद्धिमत्ता (AI) शिक्षण क्षेत्रात क्रांती घडवून आणत आहे...",
  "system": "You are a helpful multilingual assistant trained on Indic and South Asian languages."
}
```

---

## ⚡ Phase 3: Model Fine-Tuning Pipeline (`unsloth/`)

Fine-tuning is driven by [unsloth/unsloth_train_wsl.py](file:///c:/PC/Office%20Projects/Multilingual-Tokenization-Model-Integration/unsloth/unsloth_train_wsl.py) using the **Unsloth** library and **HuggingFace TRL (Transformer Reinforcement Learning)**.

### Model Architecture & Hyperparameters
- **Base Model:** `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` *(Alternative base: `bigscience/bloomz-560m`)*
- **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes` to optimize GPU memory for consumer hardware (e.g., RTX 4050 / 8GB-16GB VRAM).
- **Max Sequence Length:** 2048 tokens
- **LoRA Configuration:**
  - `r` (Rank): 16
  - `lora_alpha`: 16
  - `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - `lora_dropout`: 0 (optimized for Unsloth fast patch)
  - `bias`: `"none"`
- **Training Parameters:**
  - `per_device_train_batch_size`: 2
  - `gradient_accumulation_steps`: 4
  - `warmup_steps`: 10
  - `learning_rate`: 2e-4
  - `logging_steps`: 1
  - `optimizer`: `adamw_8bit`
  - `fp16` / `bf16`: Auto-detected based on CUDA capability (`is_bfloat16_supported()`).

### Execution Environment
The script `unsloth_train_wsl.py` is configured to run efficiently inside **WSL2 (Windows Subsystem for Linux)** to bypass Windows PyTorch DLL and C++ compilation restrictions.

---

## 💾 Phase 4: Checkpointing & Model Saving (`checkpoints/`)

During fine-tuning, model states and PEFT adapters are periodically saved to `checkpoints/`:

```
checkpoints/
├── checkpoint-120000/
├── checkpoint-122500/
└── checkpoint-125000/
    ├── adapter_config.json          # LoRA hyperparameter configuration
    ├── adapter_model.safetensors    # Trained LoRA weights
    ├── checkpoint_info.pkl          # Metadata (step, epoch, base model path)
    ├── special_tokens_map.json      # Special token mappings
    ├── tokenizer.json               # Multilingual subword tokenizer
    └── tokenizer_config.json        # Tokenizer settings
```

---

## 🧪 Phase 5: Inference Testing & Model Evaluation

After checkpoints are saved, inference scripts test the model's generation quality and script integrity:

- [test_checkpoint_125000_inference.py](file:///c:/PC/Office%20Projects/Multilingual-Tokenization-Model-Integration/test_checkpoint_125000_inference.py): Loads `checkpoint-125000` via `PeftModel` and `AutoModelForCausalLM`, then runs targeted prompts.
- [test_model_simple.py](file:///c:/PC/Office%20Projects/Multilingual-Tokenization-Model-Integration/test_model_simple.py): Lightweight inference test using `checkpoint_info.pkl` to automatically retrieve the base model and adapter paths.

### Inference Setup Example
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "unsloth/llama-3.2-3b-instruct-bnb-4bit"
adapter_path = "checkpoints/checkpoint-125000"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
```

---

## 🔄 Phase 6: Reinforcement Learning & KSML Alignment

To continuously adapt and align model responses, the platform integrates a two-stage feedback system:

### 1. KSML Semantic Alignment Engine ([sovereign_core/ksml/aligner.py](file:///c:/PC/Office%20Projects/Multilingual-Tokenization-Model-Integration/sovereign_core/ksml/aligner.py))
- Evaluates output text against **KSML** (Knowledge, Semantic, Multilingual, Language) rules.
- Tags responses with:
  - **Intent**: Informational, instructional, creative, query.
  - **Karma State**: Positive, neutral, constructive.
  - **Sanskrit Root Alignment**: Matches language tokens to root concepts in `ksml_roots.json`.

### 2. Reinforcement Learning Retraining ([reinforcement/retrain_rl.py](file:///c:/PC/Office%20Projects/Multilingual-Tokenization-Model-Integration/reinforcement/retrain_rl.py))
- **Replay Buffer**: Stores user feedback, response rewards, and agent performance (`replay_buffer.py`).
- **UCB Agent Selector**: Multi-Armed Bandit algorithm (Upper Confidence Bound) optimizes model/agent selection based on task type.
- **Automated Retraining**: When >50 new samples accumulate in the replay buffer, `RLRetrainer.retrain()` executes policy updates and writes policy snapshots (`logs/rl_backups`).

---

## 🚀 How to Run the Training & Inference Pipeline

### 1. Clean Raw Corpora
```bash
python sovereign_data_cleaning/clean_hindi.py
python sovereign_data_cleaning/clean_marathi.py
python sovereign_data_cleaning/clean_bengali.py
```

### 2. Launch Unsloth Training (in WSL)
```bash
wsl
source venv_wsl/bin/activate
export PYTHONPATH=.
python unsloth/unsloth_train_wsl.py
```

### 3. Run Inference on Trained Checkpoint
```bash
python test_checkpoint_125000_inference.py
```

### 4. Trigger RL Retraining Loop
```bash
python reinforcement/retrain_rl.py --force
```

---

## 📊 Hardware & System Requirements

| Resource | Minimum Requirement | Recommended Specification |
|:---|:---|:---|
| **GPU VRAM** | 6 GB (RTX 3060 / 4050) | 12 GB - 24 GB (RTX 4070 / 4090 / A6000) |
| **RAM** | 16 GB | 32 GB+ |
| **Storage** | 20 GB free space | SSD with 50 GB+ free space |
| **OS Environment** | Windows 11 + WSL2 (Ubuntu 22.04) | Ubuntu 22.04 LTS native |
| **Python** | 3.10 / 3.11 | 3.11 |
