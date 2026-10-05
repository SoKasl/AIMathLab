import random
import os
from generate_hierarchical_curriculum import (
    generate_addition_reasoning,
    generate_subtraction_reasoning,
    generate_multi_digit_mul,
    generate_digit_by_digit_division,
    generate_single_digit_mul
)

def generate_benchmark_4000(output_file="math_eval_4000.txt", seed=42):
    random.seed(seed)
    print(f"Generating 4,000-problem balanced evaluation benchmark (seed={seed})...")
    
    samples = []
    
    # 1. Addition (1,000 problems)
    print("  Generating 1,000 Addition problems...")
    for _ in range(500):
        # 2-3 digit
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        samples.append(generate_addition_reasoning(a, b))
    for _ in range(500):
        # 3-4 digit
        a = random.randint(100, 4999)
        b = random.randint(100, 4999)
        samples.append(generate_addition_reasoning(a, b))
        
    # 2. Subtraction (1,000 problems)
    print("  Generating 1,000 Subtraction problems...")
    for _ in range(500):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        if a < b: a, b = b, a
        samples.append(generate_subtraction_reasoning(a, b))
    for _ in range(500):
        a = random.randint(100, 4999)
        b = random.randint(100, 4999)
        if a < b: a, b = b, a
        samples.append(generate_subtraction_reasoning(a, b))
        
    # 3. Multiplication (1,000 problems)
    print("  Generating 1,000 Multiplication problems...")
    for _ in range(250):
        # 1-digit by 1-digit / 2-digit
        a = random.randint(2, 99)
        b = random.randint(2, 9)
        samples.append(generate_multi_digit_mul(a, b))
    for _ in range(500):
        # 2-digit by 2-digit
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        samples.append(generate_multi_digit_mul(a, b))
    for _ in range(250):
        # 3-digit by 2-digit
        a = random.randint(100, 999)
        b = random.randint(10, 49)
        samples.append(generate_multi_digit_mul(a, b))
        
    # 4. Division (1,000 problems)
    print("  Generating 1,000 Division problems...")
    for _ in range(300):
        # Single-digit divisor (b in 2..9, quotient in 10..999)
        b = random.randint(2, 9)
        c = random.randint(10, 999)
        a = b * c
        samples.append(generate_digit_by_digit_division(a, b))
    for _ in range(400):
        # 2-digit divisor (b in 10..49, quotient in 10..99)
        b = random.randint(10, 49)
        c = random.randint(10, 99)
        a = b * c
        samples.append(generate_digit_by_digit_division(a, b))
    for _ in range(300):
        # 2-digit divisor (b in 11..99, quotient in 10..999)
        b = random.randint(11, 99)
        c = random.randint(10, 499)
        a = b * c
        samples.append(generate_digit_by_digit_division(a, b))
        
    # Shuffle to interleave all four operations randomly
    random.shuffle(samples)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s)
            
    print(f"Successfully wrote {len(samples)} problems to {output_file} ({os.path.getsize(output_file) / 1024:.1f} KB)")

if __name__ == "__main__":
    generate_benchmark_4000()
