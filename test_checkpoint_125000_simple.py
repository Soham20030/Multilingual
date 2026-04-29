#!/usr/bin/env python3
"""
Simplified test script for checkpoint-125000 using the pickle file.
This version loads the pickle file and shows its contents, then attempts inference.
"""

import sys
import pickle
from pathlib import Path

def load_and_show_pickle_info():
    """Load and display checkpoint info from pickle file"""
    pickle_path = "checkpoints/checkpoint-125000/checkpoint_info.pkl"
    print("="*70)
    print("Loading Checkpoint Info from Pickle File")
    print("="*70)
    print(f"\n📦 Pickle file: {pickle_path}")
    
    try:
        with open(pickle_path, 'rb') as f:
            checkpoint_info = pickle.load(f)
        
        print("\n✅ Successfully loaded checkpoint info!")
        print("\n📋 Checkpoint Information:")
        print("-"*70)
        for key, value in checkpoint_info.items():
            print(f"  {key}: {value}")
        print("-"*70)
        return checkpoint_info
    except Exception as e:
        print(f"\n❌ Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()
        return None

def attempt_inference(checkpoint_info):
    """Attempt to load model and perform inference"""
    print("\n" + "="*70)
    print("Attempting Model Loading and Inference")
    print("="*70)
    
    try:
        # Import here to catch import errors
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
        print("\n✅ Successfully imported required libraries")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nPlease install required packages:")
        print("  pip install transformers peft torch")
        return False
    except Exception as e:
        print(f"\n❌ Error importing libraries: {e}")
        print("\n⚠️  This might be a PyTorch DLL issue on Windows.")
        print("   Common solutions:")
        print("   1. Restart your terminal/IDE")
        print("   2. Reinstall PyTorch: pip uninstall torch && pip install torch")
        print("   3. Use WSL if available")
        return False
    
    checkpoint_path = checkpoint_info.get('checkpoint_path', 'checkpoints/checkpoint-125000')
    base_model_name = checkpoint_info.get('base_model_name', 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit')
    
    print(f"\n📥 Loading tokenizer from: {checkpoint_path}")
    try:
        # Load tokenizer
        if Path(checkpoint_path) / "tokenizer.json" in list(Path(checkpoint_path).glob("tokenizer.*")):
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("   ✓ Tokenizer loaded")
    except Exception as e:
        print(f"   ❌ Error loading tokenizer: {e}")
        return False
    
    print(f"\n📥 Loading base model: {base_model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("   ✓ Base model loaded")
    except Exception as e:
        print(f"   ❌ Error loading base model: {e}")
        return False
    
    print(f"\n📥 Loading LoRA adapter from: {checkpoint_path}")
    try:
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model.eval()
        print("   ✓ Adapter loaded")
    except Exception as e:
        print(f"   ❌ Error loading adapter: {e}")
        return False
    
    # Now generate Arabic text
    print("\n" + "="*70)
    print("Generating Arabic Text (~500 characters)")
    print("="*70)
    
    # Format Arabic prompt
    system_message = "أنت مساعد ذكي باللغة العربية. تفهم عدة لغات (الإنجليزية، العربية، إلخ) لكن يجب أن ترد دائمًا بالعربية الفصحى فقط، بغض النظر عن لغة المدخلات. لا تستخدم الإنجليزية أو أي لغة أخرى في ردك."
    user_prompt = "اكتب مقالة قصيرة باللغة العربية عن موضوع الذكاء الاصطناعي وتأثيره على التعليم. يجب أن تكون المقالة حوالي 500 حرف."
    
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"<|system|>\n{system_message}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    
    print(f"\n📝 Prompt (Arabic): {user_prompt}")
    print("\n⏳ Generating response...")
    
    try:
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,  # More tokens to ensure ~500 characters
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|assistant|>" in full_text:
            response = full_text.split("<|assistant|>")[-1].strip()
        elif full_text.startswith(formatted_prompt):
            response = full_text[len(formatted_prompt):].strip()
        else:
            response = full_text
        response = response.split("<|user|>")[0].strip()
        response = response.split("<|system|>")[0].strip()
        
        print("\n" + "="*70)
        print("📄 Generated Arabic Text:")
        print("="*70)
        print(response)
        print("="*70)
        
        char_count = len(response)
        print(f"\n📊 Statistics:")
        print(f"   Character count: {char_count}")
        print(f"   Target: ~500 characters")
        print(f"   Difference: {abs(500 - char_count)} characters")
        
        if char_count < 400:
            print(f"\n⚠️  Generated text is shorter than expected. Regenerating with more tokens...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=450,  # Even more tokens
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "<|assistant|>" in full_text:
                response = full_text.split("<|assistant|>")[-1].strip()
            elif full_text.startswith(formatted_prompt):
                response = full_text[len(formatted_prompt):].strip()
            else:
                response = full_text
            response = response.split("<|user|>")[0].strip()
            response = response.split("<|system|>")[0].strip()
            
            char_count = len(response)
            print(f"\n📄 Regenerated Arabic Text ({char_count} characters):")
            print("="*70)
            print(response)
            print("="*70)
        
        print("\n" + "="*70)
        print("✅ Inference test complete!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Load pickle info first
    checkpoint_info = load_and_show_pickle_info()
    
    if checkpoint_info is None:
        print("\n❌ Cannot proceed without checkpoint info")
        sys.exit(1)
    
    # Attempt inference
    success = attempt_inference(checkpoint_info)
    
    if not success:
        print("\n⚠️  Inference failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

