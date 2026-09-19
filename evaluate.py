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

def evaluate_model(model=None, tokenizer=None, device=None):
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
    
    with open('math_eval_100.txt', 'r', encoding='utf-8') as f:
        content = f.read().split('\n\n')
        
    results = {'+': {'correct': 0, 'total': 0},
               '-': {'correct': 0, 'total': 0},
               '*': {'correct': 0, 'total': 0},
               '/': {'correct': 0, 'total': 0}}
               
    print(f"Starting evaluation of 100 equations...")
    
    count = 0
    for block in content:
        if not block.strip(): continue
        
        prompt, expected, op = parse_equation_block(block)
        if not prompt: continue
        
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate up to 100 tokens (enough to finish thought process)
        with torch.no_grad():
            context = model.generate_next_token(context, max_new_tokens=100)
                    
        generated_text = tokenizer.decode(context[0].tolist())
        
        # Parse model output
        # Look for the last number after </thought> or just the last number
        model_answer = None
        match = re.search(r'</thought>\s*(-?\d+)', generated_text[len(prompt):])
        if match:
            model_answer = match.group(1)
        else:
            # Fallback: just grab the last number in the generated text
            numbers = re.findall(r'-?\d+', generated_text[len(prompt):])
            if numbers:
                model_answer = numbers[-1]
                
        is_correct = (model_answer == expected)
        
        results[op]['total'] += 1
        if is_correct:
            results[op]['correct'] += 1
            
        count += 1
        if count % 10 == 0:
            print(f"Evaluated {count}/100...")
            
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
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
        print(f"Operation [{op}]: {c}/{t} ({acc:.1f}%)")
        
    overall = (total_correct/total_q)*100 if total_q > 0 else 0
    accuracies['overall'] = overall
    print("-" * 30)
    print(f"OVERALL ACCURACY: {total_correct}/{total_q} ({overall:.1f}%)")
    print("=" * 30)
    
    return accuracies

if __name__ == '__main__':
    evaluate_model()
