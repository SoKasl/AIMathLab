import random
import os

def generate_addition_reasoning(a, b):
    """ Generates 2-to-3 digit addition with place-value decomposition """
    a_100, a_10, a_1 = (a // 100) * 100, ((a // 10) % 10) * 10, a % 10
    b_100, b_10, b_1 = (b // 100) * 100, ((b // 10) % 10) * 10, b % 10
    
    parts_a = []
    if a_100 > 0: parts_a.append(str(a_100))
    if a_10 > 0: parts_a.append(str(a_10))
    if a_1 > 0 or len(parts_a) == 0: parts_a.append(str(a_1))
    
    parts_b = []
    if b_100 > 0: parts_b.append(str(b_100))
    if b_10 > 0: parts_b.append(str(b_10))
    if b_1 > 0 or len(parts_b) == 0: parts_b.append(str(b_1))
    
    step1 = f"{a_100} + {b_100} + {a_10} + {b_10} + {a_1} + {b_1}"
    step2 = f"{a_100 + b_100} + {a_10 + b_10} + {a_1 + b_1}"
    ans = a + b
    
    return f"Q: What is {a} + {b}?\nA: <thought> {a} + {b} = {step1} = {step2} = {ans} </thought> {ans}\n\n"

def generate_subtraction_reasoning(a, b):
    """ Generates multi-digit subtraction with borrow state tracking """
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

def generate_single_digit_mul(a, b):
    """ Single digit multiplication primitive (0-9 x 0-9) """
    ans = a * b
    return f"Q: What is {a} * {b}?\nA: <thought> Primitive: {a} * {b} = {ans} </thought> {ans}\n\n"

def get_base_and_zeros(n):
    if n == 0: return 0, 0
    s = str(n)
    zeros = len(s) - len(s.rstrip('0'))
    base = int(s.rstrip('0'))
    return base, zeros

def generate_multi_digit_mul(a, b):
    """ Multi-digit multiplication using FOIL with explicit zero-padding """
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
    if len(a_parts) <= 1 and len(b_parts) <= 1:
        base_a, z_a = get_base_and_zeros(a)
        base_b, z_b = get_base_and_zeros(b)
        z_tot = z_a + z_b
        val_base = base_a * base_b
        if z_tot > 0:
            return f"Q: What is {a} * {b}?\nA: <thought> {base_a} * {base_b} = {val_base}. Zeros: {z_a} + {z_b} = {z_tot}. Append {z_tot} zeros -> {ans}. </thought> {ans}\n\n"
        else:
            return f"Q: What is {a} * {b}?\nA: <thought> Primitive: {a} * {b} = {ans} </thought> {ans}\n\n"
        
    thought_lines = [f"{a} * {b} = ({a_str}) * ({b_str})"]
    foil_vals = []
    for p_a in a_parts:
        for p_b in b_parts:
            if p_a == 0 or p_b == 0: continue
            base_a, z_a = get_base_and_zeros(p_a)
            base_b, z_b = get_base_and_zeros(p_b)
            z_tot = z_a + z_b
            val_base = base_a * base_b
            val = p_a * p_b
            foil_vals.append(val)
            if z_tot > 0:
                thought_lines.append(f"{p_a}*{p_b}: {base_a}*{base_b}={val_base} with {z_tot} zeros -> {val}")
            else:
                thought_lines.append(f"{p_a}*{p_b}: {val}")
                
    foil_joined = " + ".join(str(v) for v in foil_vals)
    thought_lines.append(f"Sum: {foil_joined} = {ans}")
    thought = ". ".join(thought_lines)
    
    return f"Q: What is {a} * {b}?\nA: <thought> {thought}. </thought> {ans}\n\n"

def generate_digit_by_digit_division(a, b):
    """ Standard long division algorithm digit-by-digit """
    c = a // b
    digits = [int(d) for d in str(a)]
    
    steps = []
    quotient_digits = []
    
    curr = 0
    idx = 0
    
    # Grab initial digits until >= b
    while curr < b and idx < len(digits):
        curr = curr * 10 + digits[idx]
        idx += 1
        
    q_digit = curr // b
    prod = b * q_digit
    rem = curr - prod
    quotient_digits.append(str(q_digit))
    steps.append(f"Take {curr}: {curr} / {b} = {q_digit} ({b} * {q_digit} = {prod}), rem = {curr} - {prod} = {rem}")
    
    while idx < len(digits):
        next_d = digits[idx]
        idx += 1
        curr = rem * 10 + next_d
        q_digit = curr // b
        prod = b * q_digit
        rem = curr - prod
        quotient_digits.append(str(q_digit))
        steps.append(f"Bring down {next_d} -> {curr}: {curr} / {b} = {q_digit} ({b} * {q_digit} = {prod}), rem = {curr} - {prod} = {rem}")
        
    steps_str = ". ".join(steps)
    full_q = "".join(quotient_digits)
    thought = f"Divide {a} by {b}: {steps_str}. Final quotient = {full_q}"
    return f"Q: What is {a} / {b}?\nA: <thought> {thought}. </thought> {c}\n\n"

def build_stage_1(count=30000):
    """ Stage 1: 50% Add, 50% Sub """
    print(f"Generating Stage 1: Additive Foundations ({count} samples)...")
    samples = []
    for _ in range(count // 2):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        samples.append(generate_addition_reasoning(a, b))
        
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        if a < b: a, b = b, a
        samples.append(generate_subtraction_reasoning(a, b))
    random.shuffle(samples)
    return samples

def build_stage_2(count=30000):
    """ Stage 2: 30% Add, 30% Sub, 40% 1-Digit Mul (100% coverage of 0-9 table) """
    print(f"Generating Stage 2: Single-Digit Multiplication ({count} samples)...")
    samples = []
    # Exhaustive single-digit multiplication table
    all_single_digit = []
    for a in range(10):
        for b in range(10):
            all_single_digit.append(generate_single_digit_mul(a, b))
            
    num_mul = int(count * 0.40)
    for _ in range(num_mul):
        samples.append(random.choice(all_single_digit))
        
    num_add = int(count * 0.30)
    for _ in range(num_add):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        samples.append(generate_addition_reasoning(a, b))
        
    num_sub = count - len(samples)
    for _ in range(num_sub):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        if a < b: a, b = b, a
        samples.append(generate_subtraction_reasoning(a, b))
        
    random.shuffle(samples)
    return samples

def build_stage_3(count=40000):
    """ Stage 3: 20% Add, 20% Sub, 60% Multi-Digit Multiplication """
    print(f"Generating Stage 3: Multi-Digit Multiplication with Zero-Padding ({count} samples)...")
    samples = []
    num_mul = int(count * 0.60)
    for _ in range(num_mul):
        # Mix 2x1, 2x2, and 3x2/3x3
        r = random.random()
        if r < 0.2:
            a = random.randint(10, 99)
            b = random.randint(2, 9)
        elif r < 0.6:
            a = random.randint(10, 99)
            b = random.randint(10, 99)
        else:
            a = random.randint(100, 999)
            b = random.randint(10, 999)
        samples.append(generate_multi_digit_mul(a, b))
        
    num_add = int(count * 0.20)
    for _ in range(num_add):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        samples.append(generate_addition_reasoning(a, b))
        
    num_sub = count - len(samples)
    for _ in range(num_sub):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        if a < b: a, b = b, a
        samples.append(generate_subtraction_reasoning(a, b))
        
    random.shuffle(samples)
    return samples

def build_stage_4(count=50000):
    """ Stage 4: 10% Add, 10% Sub, 35% Mul, 45% Long Division """
    print(f"Generating Stage 4: Long Division Integration ({count} samples)...")
    samples = []
    num_div = int(count * 0.45)
    for _ in range(num_div):
        # Mix 1-digit divisors (70%) and 2-digit divisors (30%)
        if random.random() < 0.70:
            b = random.randint(2, 9)
            c = random.randint(10, 999) # Quotient
        else:
            b = random.randint(10, 99)
            c = random.randint(10, 99)
        a = b * c # Ensure clean division
        samples.append(generate_digit_by_digit_division(a, b))
        
    num_mul = int(count * 0.35)
    for _ in range(num_mul):
        a = random.randint(10, 999)
        b = random.randint(2, 999)
        samples.append(generate_multi_digit_mul(a, b))
        
    num_add = int(count * 0.10)
    for _ in range(num_add):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        samples.append(generate_addition_reasoning(a, b))
        
    num_sub = count - len(samples)
    for _ in range(num_sub):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        if a < b: a, b = b, a
        samples.append(generate_subtraction_reasoning(a, b))
        
    random.shuffle(samples)
    return samples

def build_eval_set(count=100):
    """ Balanced evaluation set: 25 per operation """
    print(f"Generating Balanced Hierarchical Evaluation Set ({count} samples)...")
    samples = []
    # 25 Addition
    for _ in range(25):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        samples.append(generate_addition_reasoning(a, b))
    # 25 Subtraction
    for _ in range(25):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        if a < b: a, b = b, a
        samples.append(generate_subtraction_reasoning(a, b))
    # 25 Multiplication
    for _ in range(25):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        samples.append(generate_multi_digit_mul(a, b))
    # 25 Division
    for _ in range(25):
        if random.random() < 0.6:
            b = random.randint(2, 9)
            c = random.randint(50, 999)
        else:
            b = random.randint(11, 49)
            c = random.randint(11, 99)
        a = b * c
        samples.append(generate_digit_by_digit_division(a, b))
        
    return samples

def main():
    random.seed(42)
    stage1 = build_stage_1(30000)
    with open("math_stage1_train.txt", "w", encoding="utf-8") as f:
        f.writelines(stage1)
        
    stage2 = build_stage_2(30000)
    with open("math_stage2_train.txt", "w", encoding="utf-8") as f:
        f.writelines(stage2)
        
    stage3 = build_stage_3(40000)
    with open("math_stage3_train.txt", "w", encoding="utf-8") as f:
        f.writelines(stage3)
        
    stage4 = build_stage_4(50000)
    with open("math_stage4_train.txt", "w", encoding="utf-8") as f:
        f.writelines(stage4)
        
    eval_set = build_eval_set(100)
    with open("math_eval_hierarchical.txt", "w", encoding="utf-8") as f:
        f.writelines(eval_set)
        
    print("All hierarchical curriculum datasets generated successfully!")

if __name__ == "__main__":
    main()
