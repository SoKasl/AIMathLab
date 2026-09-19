import random
import os

def generate_addition_reasoning(a, b):
    """ Generates 3-digit addition with step-by-step reasoning """
    # e.g., 123 + 456
    a_100, a_10, a_1 = (a // 100) * 100, ((a // 10) % 10) * 10, a % 10
    b_100, b_10, b_1 = (b // 100) * 100, ((b // 10) % 10) * 10, b % 10
    
    step1 = f"{a_100} + {b_100} + {a_10} + {b_10} + {a_1} + {b_1}"
    step2 = f"{a_100 + b_100} + {a_10 + b_10} + {a_1 + b_1}"
    ans = a + b
    
    return f"Q: What is {a} + {b}?\nA: <thought> {a} + {b} = {step1} = {step2} </thought> {ans}\n\n"

def generate_subtraction_reasoning(a, b):
    """ Generates 3-digit subtraction with step-by-step reasoning (a > b) """
    a_100, a_10, a_1 = (a // 100) * 100, ((a // 10) % 10) * 10, a % 10
    b_100, b_10, b_1 = (b // 100) * 100, ((b // 10) % 10) * 10, b % 10
    
    step1 = f"{a_100} - {b_100} + {a_10} - {b_10} + {a_1} - {b_1}"
    step2 = f"{a_100 - b_100} + {a_10 - b_10} + {a_1 - b_1}"
    ans = a - b
    
    return f"Q: What is {a} - {b}?\nA: <thought> {a} - {b} = {step1} = {step2} </thought> {ans}\n\n"

def generate_simple_equation(a, b, op):
    """ Standard output for multiplication and division to keep it simple """
    if op == '*':
        return f"Q: What is {a} * {b}?\nA: <thought> Calculating {a} times {b} </thought> {a * b}\n\n"
    elif op == '/':
        return f"Q: What is {a} / {b}?\nA: <thought> Calculating {a} divided by {b} </thought> {a // b}\n\n"

def generate_datasets():
    used_equations = set()
    train_data = []
    eval_data = []
    
    print("Generating Training Data (100,000 samples)...")
    while len(train_data) < 100000:
        op = random.choice(['+', '-', '*', '/'])
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        
        if op == '-':
            if a < b: a, b = b, a # Ensure positive
        elif op == '/':
            a = a * b # Ensure clean division
            
        eq_key = f"{a}{op}{b}"
        if eq_key in used_equations:
            continue
            
        used_equations.add(eq_key)
        
        if op == '+':
            text = generate_addition_reasoning(a, b)
        elif op == '-':
            text = generate_subtraction_reasoning(a, b)
        else:
            text = generate_simple_equation(a, b, op)
            
        train_data.append(text)
        
    print("Generating Eval Data (5,000 samples with ZERO overlap)...")
    while len(eval_data) < 5000:
        op = random.choice(['+', '-', '*', '/'])
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        
        if op == '-':
            if a < b: a, b = b, a
        elif op == '/':
            a = a * b
            
        eq_key = f"{a}{op}{b}"
        if eq_key in used_equations:
            continue # STRICT LEAKAGE PREVENTION
            
        used_equations.add(eq_key)
        
        if op == '+':
            text = generate_addition_reasoning(a, b)
        elif op == '-':
            text = generate_subtraction_reasoning(a, b)
        else:
            text = generate_simple_equation(a, b, op)
            
        eval_data.append(text)
        
    with open("math_train.txt", "w", encoding="utf-8") as f:
        f.writelines(train_data)
    with open("math_eval.txt", "w", encoding="utf-8") as f:
        f.writelines(eval_data)
        
    print("Data successfully generated and persisted!")
    print(f"Total unique equations generated: {len(used_equations)}")

if __name__ == "__main__":
    generate_datasets()
