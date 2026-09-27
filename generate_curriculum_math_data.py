import random
import os

def generate_addition_reasoning(a, b):
    """ Generates 3-digit addition with step-by-step reasoning """
    a_100, a_10, a_1 = (a // 100) * 100, ((a // 10) % 10) * 10, a % 10
    b_100, b_10, b_1 = (b // 100) * 100, ((b // 10) % 10) * 10, b % 10
    
    step1 = f"{a_100} + {b_100} + {a_10} + {b_10} + {a_1} + {b_1}"
    step2 = f"{a_100 + b_100} + {a_10 + b_10} + {a_1 + b_1}"
    ans = a + b
    
    return f"Q: What is {a} + {b}?\nA: <thought> {a} + {b} = {step1} = {step2} </thought> {ans}\n\n"

def generate_subtraction_reasoning(a, b):
    """ Generates 3-digit subtraction with human-like borrowing """
    ans = a - b
    
    a_100, a_10, a_1 = a // 100, (a // 10) % 10, a % 10
    b_100, b_10, b_1 = b // 100, (b // 10) % 10, b % 10
    
    ones_step = f"{a_1} ones - {b_1} ones"
    if a_1 < b_1:
        ones_step += f" (borrow 10) -> {a_1 + 10} - {b_1} = {a_1 + 10 - b_1}"
        a_10 -= 1
        val_1 = a_1 + 10 - b_1
    else:
        ones_step += f" -> {a_1} - {b_1} = {a_1 - b_1}"
        val_1 = a_1 - b_1
        
    tens_step = f"{a_10} tens - {b_10} tens"
    if a_10 < b_10:
        tens_step += f" (borrow 100) -> {a_10 * 10 + 100} - {b_10 * 10} = {a_10 * 10 + 100 - b_10 * 10}"
        a_100 -= 1
        val_10 = a_10 * 10 + 100 - b_10 * 10
    else:
        tens_step += f" -> {a_10 * 10} - {b_10 * 10} = {a_10 * 10 - b_10 * 10}"
        val_10 = a_10 * 10 - b_10 * 10
        
    hundreds_step = f"{a_100} hundreds - {b_100} hundreds -> {a_100 * 100} - {b_100 * 100} = {a_100 * 100 - b_100 * 100}"
    val_100 = a_100 * 100 - b_100 * 100
    
    return f"Q: What is {a} - {b}?\nA: <thought> {ones_step}. {tens_step}. {hundreds_step}. Answer = {val_100} + {val_10} + {val_1} = {ans} </thought> {ans}\n\n"

def get_base_and_zeros(n):
    if n == 0: return 0, 0
    s = str(n)
    zeros = len(s) - len(s.rstrip('0'))
    base = int(s.rstrip('0'))
    return base, zeros

def generate_multiplication_reasoning(a, b):
    """ Generates 3-digit multiplication using the Grid (FOIL) method with Curriculum Shift """
    a_parts = []
    if a // 100 > 0: a_parts.append((a // 100) * 100)
    if (a // 10) % 10 > 0: a_parts.append(((a // 10) % 10) * 10)
    if a % 10 > 0: a_parts.append(a % 10)
    
    b_parts = []
    if b // 100 > 0: b_parts.append((b // 100) * 100)
    if (b // 10) % 10 > 0: b_parts.append(((b // 10) % 10) * 10)
    if b % 10 > 0: b_parts.append(b % 10)
    
    a_str = " + ".join(str(p) for p in a_parts) if len(a_parts) > 1 else str(a)
    b_str = " + ".join(str(p) for p in b_parts) if len(b_parts) > 1 else str(b)
    
    ans = a * b
    
    if len(a_parts) == 1 and len(b_parts) == 1:
        return f"Q: What is {a} * {b}?\nA: <thought> {a} * {b} = {ans} </thought> {ans}\n\n"
        
    thought = f"\n{a} * {b} = ({a_str}) * ({b_str})\n"
    
    foil_vals = []
    for p_a in a_parts:
        for p_b in b_parts:
            if p_a == 0 or p_b == 0:
                continue
                
            base_a, z_a = get_base_and_zeros(p_a)
            base_b, z_b = get_base_and_zeros(p_b)
            z_total = z_a + z_b
            val_base = base_a * base_b
            val = p_a * p_b
            foil_vals.append(val)
            
            if z_total > 0:
                thought += f"{p_a}*{p_b}=cl:{base_a}*{base_b}e{z_total}={val_base}e{z_total}={val}\n"
            else:
                thought += f"{p_a}*{p_b}=cl:{base_a}*{base_b}={val}\n"
                
    foil_vals_joined = " + ".join(str(v) for v in foil_vals)
    thought += f"Sum = {foil_vals_joined} = {ans}\n"
    
    return f"Q: What is {a} * {b}?\nA: <thought> {thought}</thought> {ans}\n\n"

def generate_division_reasoning(a, b):
    """ Generates division using partial quotients """
    c = a // b
    c_parts = []
    if c // 100 > 0: c_parts.append((c // 100) * 100)
    if (c // 10) % 10 > 0: c_parts.append(((c // 10) % 10) * 10)
    if c % 10 > 0: c_parts.append(c % 10)
    
    steps = []
    current_rem = a
    for p in c_parts:
        prod = b * p
        current_rem -= prod
        steps.append(f"{b} * {p} = {prod}, remaining = {current_rem}")
    
    steps_str = ". ".join(steps)
    ans_sum_str = " + ".join(str(p) for p in c_parts) if len(c_parts) > 1 else str(c)
    
    return f"Q: What is {a} / {b}?\nA: <thought> {a} / {b}: {steps_str}. Answer = {ans_sum_str} = {c} </thought> {c}\n\n"

def generate_datasets():
    used_equations = set()
    train_data = []
    eval_data = []

    print("Injecting Curriculum Learning: Single Digit Multiplication & Division...")
    for a in range(1, 10):
        for b in range(1, 10):
            ans_m = a * b
            text_m = f"Q: What is {a} * {b}?\nA: <thought> cl: {a} * {b} = {ans_m} </thought> {ans_m}\n\n"
            for _ in range(50): train_data.append(text_m)
            used_equations.add(f"{a}*{b}")
            
            ans_d = a * b
            text_d = f"Q: What is {ans_d} / {a}?\nA: <thought> cl: {ans_d} / {a} = {b} </thought> {b}\n\n"
            for _ in range(50): train_data.append(text_d)
            used_equations.add(f"{ans_d}/{a}")
            
    
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
        elif op == '*':
            text = generate_multiplication_reasoning(a, b)
        elif op == '/':
            text = generate_division_reasoning(a, b)
            
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
        elif op == '*':
            text = generate_multiplication_reasoning(a, b)
        elif op == '/':
            text = generate_division_reasoning(a, b)
            
        eval_data.append(text)
        
    with open("math_train_curriculum.txt", "w", encoding="utf-8") as f:
        f.writelines(train_data)
    with open("math_eval_curriculum.txt", "w", encoding="utf-8") as f:
        f.writelines(eval_data)
        
    print("Data successfully generated and persisted!")
    print(f"Total unique equations generated: {len(used_equations)}")

if __name__ == "__main__":
    generate_datasets()
