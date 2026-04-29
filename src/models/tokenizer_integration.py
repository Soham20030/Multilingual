import sentencepiece as spm
from transformers import AutoModelForMaskedLM, AlbertTokenizer
import torch

# Load SentencePiece tokenizer using SentencePiece library
sp = spm.SentencePieceProcessor()
sp.Load("multi_tokenizer.model")

# Alternatively, you can create a tokenizer that wraps around SentencePiece:
from tokenizers import SentencePieceBPETokenizer

tokenizer = SentencePieceBPETokenizer("multi_tokenizer.model")

# Load base multilingual model - use appropriate model for SentencePiece tokenizer
model_name = "ai4bharat/indic-bert"
model = AutoModelForMaskedLM.from_pretrained(model_name)

print("Tokenizer and base multilingual model loaded successfully.")

# Example sentences in multiple languages
sample_sentences = [
    "यह एक परीक्षण वाक्य है।",          # Hindi
    "हे एक चाचणी वाक्य आहे.",           # Marathi
    "इयं परीक्षणा वाक्यं अस्ति।",       # Sanskrit
    "This is a test sentence."           # English
]

for sent in sample_sentences:
    # Encode using SentencePieceProcessor directly
    tokens_ids = sp.EncodeAsIds(sent)
    tokens = sp.EncodeAsPieces(sent)
    print(f"\nOriginal Text: {sent}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {tokens_ids}")

    # Convert tokens_ids to torch tensors and expand batch dimension
    input_ids = torch.tensor([tokens_ids])

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        print(f"Logits shape: {logits.shape}")

print("\nIntegration and test complete.")
