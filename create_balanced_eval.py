import os

def create_balanced_eval():
    counts = {'+': 0, '-': 0, '*': 0, '/': 0}
    selected_lines = []
    
    with open('math_eval.txt', 'r', encoding='utf-8') as f:
        content = f.read().split('\n\n')
        
    for block in content:
        if not block.strip():
            continue
        
        # Determine operation
        # The block looks like:
        # Q: What is A + B?
        # A: <thought> ... </thought> C
        
        lines = block.strip().split('\n')
        if len(lines) < 2: continue
        
        question = lines[0]
        op = None
        if '+' in question: op = '+'
        elif '-' in question: op = '-'
        elif '*' in question: op = '*'
        elif '/' in question: op = '/'
        
        if op and counts[op] < 25:
            selected_lines.append(block.strip() + '\n\n')
            counts[op] += 1
            
        if all(c == 25 for c in counts.values()):
            break
            
    with open('math_eval_100.txt', 'w', encoding='utf-8') as f:
        f.writelines(selected_lines)
        
    print(f"Created math_eval_100.txt with:")
    for op, c in counts.items():
        print(f"  {op}: {c} questions")

if __name__ == '__main__':
    create_balanced_eval()
