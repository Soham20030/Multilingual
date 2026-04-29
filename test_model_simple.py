#!/usr/bin/env python3
"""
Simple test script for the trained Arabic model using transformers/peft
Uses checkpoint-125000 pickle file for checkpoint information
"""

import pickle
import os
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def convert_windows_path_to_wsl(windows_path):
    r"""Convert Windows path (C:\Users\...) to WSL path (/mnt/c/Users/...)"""
    if not windows_path:
        return windows_path
    
    # Check if it's already a WSL/Linux path
    if windows_path.startswith('/'):
        return windows_path
    
    # Convert Windows path to WSL path
    # C:\Users\... -> /mnt/c/Users/...
    path = windows_path.replace('\\', '/')
    
    # Handle drive letters (C:, D:, etc.)
    if ':' in path:
        parts = path.split(':', 1)
        drive_letter = parts[0].lower()
        rest = parts[1].lstrip('/')
        return f'/mnt/{drive_letter}/{rest}'
    
    return path

# Load checkpoint info from pickle file
pickle_path = "checkpoints/checkpoint-125000/checkpoint_info.pkl"
print("="*70)
print("Loading checkpoint info from pickle file...")
print(f"Pickle file: {pickle_path}")

try:
    with open(pickle_path, 'rb') as f:
        checkpoint_info = pickle.load(f)
    checkpoint_path = checkpoint_info.get('checkpoint_path', 'checkpoints/checkpoint-125000')
    
    # Convert Windows path to WSL path if needed
    checkpoint_path = convert_windows_path_to_wsl(checkpoint_path)
    
    # Also check if path exists, if not try relative path
    if not os.path.exists(checkpoint_path):
        # Try relative path as fallback
        rel_path = "checkpoints/checkpoint-125000"
        if os.path.exists(rel_path):
            checkpoint_path = rel_path
    
    base_model_name = checkpoint_info.get('base_model_name', 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit')
    print(f"✓ Loaded checkpoint info:")
    print(f"  - Checkpoint path: {checkpoint_path}")
    print(f"  - Base model: {base_model_name}")
    print(f"  - Step: {checkpoint_info.get('step', 'N/A')}")
    print(f"  - Epoch: {checkpoint_info.get('epoch', 'N/A')}")
except Exception as e:
    print(f"⚠️  Could not load pickle file: {e}")
    print("  ↳ Using default values")
    checkpoint_path = "checkpoints/checkpoint-125000"
    base_model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

print("\n" + "="*70)
print("Loading base model...")
print("="*70)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load adapter
print(f"\nLoading adapter from: {checkpoint_path}...")
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()
print("✓ Model loaded successfully!")

# Arabic prompt template - Enhanced to force Arabic output
# Using Llama chat format for better instruction following
def format_prompt(user_input):
    """Format prompt using Llama chat template"""
    system_message = "أنت مساعد ذكي باللغة العربية. تفهم عدة لغات (الإنجليزية، العربية، إلخ) لكن يجب أن ترد دائمًا بالعربية الفصحى فقط، بغض النظر عن لغة المدخلات. لا تستخدم الإنجليزية أو أي لغة أخرى في ردك."
    
    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ""}
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

def generate_response(prompt, max_new_tokens=256, target_chars=None):
    """Generate response from the model
    
    Args:
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        target_chars: Target number of characters (will adjust max_new_tokens if needed)
    
    Returns:
        tuple: (response_text, generation_time_seconds)
    """
    formatted_prompt = format_prompt(prompt)
    
    # Adjust max_new_tokens based on target characters
    # Arabic typically has ~2-3 characters per token
    if target_chars:
        estimated_tokens = int(target_chars / 2.5)  # Conservative estimate
        max_new_tokens = max(max_new_tokens, estimated_tokens + 50)  # Add buffer
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to(model.device)
    
    # Start timing
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # End timing
    generation_time = time.perf_counter() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    # Try to find assistant tag first
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    elif "assistant" in response.lower():
        # Try to extract after assistant role
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
            # Remove any remaining role tags
            response = response.split("<|")[0].strip()
    elif response.startswith(formatted_prompt):
        response = response[len(formatted_prompt):].strip()
    
    # Clean up any remaining prompt artifacts
    response = response.split("<|user|>")[0].strip()
    response = response.split("<|system|>")[0].strip()
    
    return response, generation_time

def run_generation_test(target_chars, prompt_suffix):
    """Run a generation test for a specific character target"""
    print("\n" + "="*70)
    print(f"TESTING: Generate Arabic Text (~{target_chars} characters)")
    print("="*70 + "\n")
    
    arabic_prompt = f"اكتب مقالة {prompt_suffix} باللغة العربية عن موضوع الذكاء الاصطناعي وتأثيره على التعليم. يجب أن تكون المقالة حوالي {target_chars} حرف."
    english_prompt = f"Write a {prompt_suffix} article in Arabic about artificial intelligence and its impact on education. The article should be approximately {target_chars} characters long."
    
    print(f"📝 Prompt (Arabic): {arabic_prompt}")
    print(f"📝 Prompt (English): {english_prompt}")
    print("\n" + "-" * 70)
    print("Generating response...")
    print("-" * 70 + "\n")
    
    try:
        # Generate with target characters
        estimated_tokens = int(target_chars / 2.5) + 50  # Add buffer
        response, generation_time = generate_response(arabic_prompt, max_new_tokens=estimated_tokens, target_chars=target_chars)
        
        char_count = len(response)
        
        # If too short, regenerate with more tokens
        if char_count < int(target_chars * 0.8):  # If less than 80% of target
            print(f"⚠️  Generated text is shorter than expected. Regenerating with more tokens...")
            estimated_tokens = int(target_chars / 2.0) + 100  # More tokens
            response, generation_time = generate_response(arabic_prompt, max_new_tokens=estimated_tokens, target_chars=target_chars)
            char_count = len(response)
        
        print("📄 Generated Arabic Text:")
        print("="*70)
        print(response)
        print("="*70)
        
        # Calculate characters per second
        chars_per_second = char_count / generation_time if generation_time > 0 else 0
        
        print(f"\n📊 Statistics:")
        print(f"   Character count: {char_count}")
        print(f"   Target: ~{target_chars} characters")
        print(f"   Difference: {abs(target_chars - char_count)} characters")
        print(f"   Generation time: {generation_time:.2f} seconds")
        print(f"   Speed: {chars_per_second:.2f} characters/second")
        if generation_time > 0:
            tokens_per_second = estimated_tokens / generation_time
            print(f"   Estimated tokens per second: ~{tokens_per_second:.2f}")
        else:
            print(f"   Tokens per second: N/A")
        
        return {
            'target_chars': target_chars,
            'actual_chars': char_count,
            'generation_time': generation_time,
            'chars_per_second': chars_per_second,
            'response': response
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run tests for 500 and 1000 characters
results = []

# Test 1: 500 characters
result_500 = run_generation_test(500, "قصيرة")
if result_500:
    results.append(result_500)

# Test 2: 1000 characters  
result_1000 = run_generation_test(1000, "متوسطة الطول")
if result_1000:
    results.append(result_1000)

# Summary
print("\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)

if results:
    print(f"\n{'Test':<15} {'Target':<10} {'Actual':<10} {'Time (s)':<12} {'Chars/s':<12}")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        test_name = f"{result['target_chars']} chars"
        print(f"{test_name:<15} {result['target_chars']:<10} {result['actual_chars']:<10} {result['generation_time']:<12.2f} {result['chars_per_second']:<12.2f}")
    
    if len(results) == 2:
        print(f"\n⏱️  Timing Comparison:")
        print(f"   500 chars:  {results[0]['generation_time']:.2f} seconds")
        print(f"   1000 chars: {results[1]['generation_time']:.2f} seconds")
        print(f"   Ratio: {results[1]['generation_time'] / results[0]['generation_time']:.2f}x" if results[0]['generation_time'] > 0 else "   Ratio: N/A")

print("\n" + "="*70)
print("✅ Testing complete!")
print("="*70)

