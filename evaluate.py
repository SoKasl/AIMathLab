import torch
import tiktoken
import os
import re
from model import GPT

def parse_equation_block(block):
    """ Parses a block of text into prompt and expected answer """
    lines = block.strip().split('\n')
    if len(lines) < 2: return None, None, None
    
    question = lines[0] # "Q: What is A + B?"
    answer_line = lines[1] # "A: <thought> ... </thought> 123"
    
    # Extract the final number as the expected answer
    match = re.search(r'</thought>\s*(-?\d+)', answer_line)
    if not match:
        # Fallback if thought tags aren't there for some reason
        parts = answer_line.split()
        expected = parts[-1]
    else:
        expected = match.group(1)
        
    # The prompt to feed the model
    prompt = f"{question}\nA:"
    
    # Determine operation
    op = None
    if '+' in question: op = '+'
    elif '-' in question: op = '-'
    elif '*' in question: op = '*'
    elif '/' in question: op = '/'
        
    return prompt, expected, op

def evaluate_model(model=None, tokenizer=None, device=None, use_old_data=False, use_hierarchical_data=False, sample_log_path=None, eval_file=None, max_problems=None):
    if sample_log_path and os.path.exists(sample_log_path):
        os.remove(sample_log_path) # Clear previous log for this step
        
    if model is None:
        print("Initializing Standalone Evaluator...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        
        tokenizer = tiktoken.get_encoding("gpt2")
        model = GPT().to(device)
        
        if os.path.exists('model.pt'):
            model.load_state_dict(torch.load('model.pt', map_location=device))
            print("Loaded trained model.pt")
        else:
            print("WARNING: model.pt not found! Evaluating an untrained random model.")
            
        model.eval()
    else:
        # Running as part of an experiment loop
        model.eval()
    
    if eval_file is None:
        if use_hierarchical_data:
            eval_file = 'math_eval_hierarchical.txt'
        elif use_old_data:
            eval_file = 'math_eval_100_old.txt'
        else:
            eval_file = 'math_eval_100.txt'
            
    if not os.path.exists(eval_file) and os.path.exists(os.path.join("data", eval_file)):
        eval_file = os.path.join("data", eval_file)
        
    with open(eval_file, 'r', encoding='utf-8') as f:
        content = [b for b in f.read().split('\n\n') if b.strip()]
        
    if max_problems is not None and max_problems > 0:
        content = content[:max_problems]
        
    total_samples = len(content)
    print(f"Starting evaluation of {total_samples} problems from {eval_file}...")
    print(f"KV Cache Enabled: {getattr(model.config, 'use_kv_cache', False)}")
    
    results = {'+': {'correct': 0, 'total': 0},
               '-': {'correct': 0, 'total': 0},
               '*': {'correct': 0, 'total': 0},
               '/': {'correct': 0, 'total': 0}}
               
    count = 0
    start_time = os.times().elapsed
    for block in content:
        if not block.strip(): continue
        
        prompt, expected, op = parse_equation_block(block)
        if not prompt: continue
        
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate up to 256 tokens (enough to finish the long FOIL / Division thought process)
        with torch.no_grad():
            if getattr(model.config, 'use_kv_cache', False):
                context = model.generate_with_cache(context, max_new_tokens=256)
            else:
                context = model.generate_next_token(context, max_new_tokens=256)
                    
        generated_text = tokenizer.decode(context[0].tolist())
        
        # Parse model output
        model_answer = None
        match = re.search(r'</thought>\s*(-?\d+)', generated_text[len(prompt):])
        if match:
            model_answer = match.group(1)
        else:
            numbers = re.findall(r'-?\d+', generated_text[len(prompt):])
            if numbers:
                model_answer = numbers[-1]
                
        is_correct = (model_answer == expected)
        
        if sample_log_path:
            with open(sample_log_path, 'a', encoding='utf-8') as sf:
                sf.write(f"=== Equation {count+1} (Correct: {is_correct}) ===\n")
                sf.write(generated_text)
                sf.write(f"\n\nExpected: {expected} | Extracted: {model_answer}\n")
                sf.write("="*40 + "\n\n")
        
        results[op]['total'] += 1
        if is_correct:
            results[op]['correct'] += 1
            
        count += 1
        log_interval = 10 if total_samples <= 100 else (100 if total_samples <= 2000 else 250)
        if count % log_interval == 0 or count == total_samples:
            elapsed = os.times().elapsed - start_time
            rate = count / max(elapsed, 0.001)
            print(f"Evaluated {count}/{total_samples} problems ({rate:.1f} problems/sec)...")
            
    print("\n" + "="*35)
    print("EVALUATION RESULTS")
    print("="*35)
    total_correct = 0
    total_q = 0
    accuracies = {}
    for op in ['+', '-', '*', '/']:
        c = results[op]['correct']
        t = results[op]['total']
        total_correct += c
        total_q += t
        acc = (c/t)*100 if t > 0 else 0
        accuracies[op] = acc
        print(f"Operation [{op}]: {c}/{t} ({acc:.2f}%)")
        
    overall = (total_correct/total_q)*100 if total_q > 0 else 0
    accuracies['overall'] = overall
    print("-" * 35)
    print(f"OVERALL ACCURACY: {total_correct}/{total_q} ({overall:.2f}%)")
    print("=" * 35)
    
    return accuracies

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained GPT math model")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to checkpoint .pt")
    parser.add_argument("--eval_file", type=str, default=None, help="Explicit eval file path (e.g. math_eval_4000.txt)")
    parser.add_argument("--use_old_data", action="store_true", help="Use old non-CoT eval set")
    parser.add_argument("--use_hierarchical_data", action="store_true", help="Use hierarchical curriculum eval set")
    parser.add_argument("--max_problems", type=int, default=None, help="Limit number of problems to evaluate")
    parser.add_argument("--sample_log_path", type=str, default=None, help="Path to log generated samples")
    
    # Architecture configuration flags
    parser.add_argument("--flash_attention", action="store_true", help="Enable FlashAttention")
    parser.add_argument("--swiglu", action="store_true", help="Enable SwiGLU")
    parser.add_argument("--rmsnorm", action="store_true", help="Enable RMSNorm")
    parser.add_argument("--rope", action="store_true", help="Enable RoPE")
    parser.add_argument("--moe", action="store_true", help="Enable Mixture-of-Experts")
    parser.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache during evaluation")
    
    args = parser.parse_args()
    
    if args.use_kv_cache:
        from model_kv_cache import GPT, config
    else:
        from model import GPT, config
        
    config.use_flash_attention = args.flash_attention
    config.use_swiglu = args.swiglu
    config.use_rmsnorm = args.rmsnorm
    config.use_rope = args.rope
    config.use_moe = args.moe
    config.use_kv_cache = args.use_kv_cache
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT().to(device)
    
    model_path = args.model_path
    if not os.path.exists(model_path) and os.path.exists(os.path.join("checkpoints", model_path)):
        model_path = os.path.join("checkpoints", model_path)
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded checkpoint: {model_path}")
    else:
        print(f"WARNING: Checkpoint {model_path} not found! Evaluating an untrained random model.")
        
    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_old_data=args.use_old_data,
        use_hierarchical_data=args.use_hierarchical_data,
        sample_log_path=args.sample_log_path,
        eval_file=args.eval_file,
        max_problems=args.max_problems
    )
