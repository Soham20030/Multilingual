#!/usr/bin/env python3
"""
Test inference script for checkpoint-125000 using the pickle file.
Generates Arabic text output (~500 characters).
"""

import sys
import pickle
from pathlib import Path
from utils.load_checkpoint_for_inference import load_model_from_checkpoint, generate_response

def format_arabic_prompt(user_input: str, tokenizer) -> str:
    """Format prompt using Llama chat template for Arabic generation"""
    system_message = "أنت مساعد ذكي باللغة العربية. تفهم عدة لغات (الإنجليزية، العربية، إلخ) لكن يجب أن ترد دائمًا بالعربية الفصحى فقط، بغض النظر عن لغة المدخلات. لا تستخدم الإنجليزية أو أي لغة أخرى في ردك."
    
    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback to simple format
        return f"""<|system|>
{system_message}
<|user|>
{user_input}
<|assistant|>
"""

def extract_assistant_response(full_text: str, prompt: str) -> str:
    """Extract only the assistant's response from the full generated text"""
    # Try to find assistant tag first
    if "<|assistant|>" in full_text:
        response = full_text.split("<|assistant|>")[-1].strip()
    elif "assistant" in full_text.lower():
        parts = full_text.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
            response = response.split("<|")[0].strip()
    elif full_text.startswith(prompt):
        response = full_text[len(prompt):].strip()
    else:
        response = full_text
    
    # Clean up any remaining prompt artifacts
    response = response.split("<|user|>")[0].strip()
    response = response.split("<|system|>")[0].strip()
    
    return response

def main():
    print("="*70)
    print("Testing Checkpoint-125000 Inference with Pickle File")
    print("="*70)
    print()
    
    # Load checkpoint info from pickle
    pickle_path = "checkpoints/checkpoint-125000/checkpoint_info.pkl"
    print(f"📦 Loading checkpoint info from pickle: {pickle_path}")
    
    try:
        with open(pickle_path, 'rb') as f:
            checkpoint_info = pickle.load(f)
        
        print(f"   ✓ Checkpoint step: {checkpoint_info.get('step', 'N/A')}")
        print(f"   ✓ Base model: {checkpoint_info.get('base_model_name', 'N/A')}")
        print(f"   ✓ Checkpoint path: {checkpoint_info.get('checkpoint_path', 'N/A')}")
        print()
    except Exception as e:
        print(f"   ⚠️ Could not load pickle file: {e}")
        print("   ↳ Will proceed with direct checkpoint path")
        pickle_path = None
        checkpoint_info = None
        print()
    
    # Load model and tokenizer using the pickle file
    print("📥 Loading model and tokenizer...")
    try:
        if pickle_path:
            model, tokenizer = load_model_from_checkpoint(pickle_path=pickle_path)
        else:
            model, tokenizer = load_model_from_checkpoint(checkpoint_path="checkpoints/checkpoint-125000")
        print("   ✓ Model and tokenizer loaded successfully!")
        print()
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test generation with Arabic prompt requesting ~500 characters
    print("="*70)
    print("🧪 Testing Arabic Text Generation (~500 characters)")
    print("="*70)
    print()
    
    # Create a prompt that asks for Arabic text generation
    arabic_prompt = "اكتب مقالة قصيرة باللغة العربية عن موضوع الذكاء الاصطناعي وتأثيره على التعليم. يجب أن تكون المقالة حوالي 500 حرف."
    english_prompt = "Write a short article in Arabic about artificial intelligence and its impact on education. The article should be approximately 500 characters long."
    
    # Try Arabic prompt first
    print(f"📝 Prompt (Arabic): {arabic_prompt}")
    print(f"📝 Prompt (English): {english_prompt}")
    print()
    print("Generating response...")
    print("-"*70)
    
    try:
        # Format the prompt using chat template
        formatted_prompt = format_arabic_prompt(arabic_prompt, tokenizer)
        
        # Generate response - we'll need more tokens for 500 Arabic characters
        # Arabic typically has ~2-3 characters per token, so we need ~200-250 tokens
        # Let's generate more to ensure we get at least 500 characters
        response = generate_response(
            model,
            tokenizer,
            formatted_prompt,
            max_new_tokens=300,  # More tokens to ensure we get ~500 characters
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Extract only the assistant's response
        clean_response = extract_assistant_response(response, formatted_prompt)
        
        print("\n📄 Generated Arabic Text:")
        print("="*70)
        print(clean_response)
        print("="*70)
        
        # Count characters
        char_count = len(clean_response)
        print(f"\n📊 Statistics:")
        print(f"   Character count: {char_count}")
        print(f"   Target: ~500 characters")
        print(f"   Difference: {abs(500 - char_count)} characters")
        
        if char_count < 400:
            print(f"\n⚠️  Generated text is shorter than expected. Re-generating with more tokens...")
            # Generate again with more tokens
            response = generate_response(
                model,
                tokenizer,
                formatted_prompt,
                max_new_tokens=400,  # Even more tokens
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            clean_response = extract_assistant_response(response, formatted_prompt)
            char_count = len(clean_response)
            print(f"\n📄 Regenerated Arabic Text ({char_count} characters):")
            print("="*70)
            print(clean_response)
            print("="*70)
        
        print()
        print("="*70)
        print("✅ Inference test complete!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

