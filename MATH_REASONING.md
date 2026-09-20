# Mathematical Reasoning in LLMs

This document captures the evolution of our synthetic math dataset. We found that standard autoregressive language models fundamentally struggle to perform complex arithmetic (like 3-digit multiplication) in a single forward pass. 

To solve this, we introduced `<thought>` tags to force the model to "show its work" and act as a scratchpad. However, the *quality* of that scratchpad is critical.

Here is how we evolved the reasoning traces for each operation.

---

## 1. Multiplication

### ❌ Before (Fake Reasoning)
The original implementation simply printed a static string and forced the model to calculate the entire answer in a single jump.
```text
Q: What is 123 * 45?
A: <thought> Calculating 123 times 45 </thought> 5535
```
**Why it failed:** The model was still forced to predict `5535` without any intermediate steps to hold carry values or partial sums. Accuracy remained at exactly 0.0% across 25,000 steps of training.

### ✅ After (FOIL Expansion / Grid Method)
We implemented a human-like grid expansion. Both numbers are broken down by magnitude (hundreds, tens, ones). The model only ever has to execute single-digit multiplications (from the 10x10 times table) and append zeros.
```text
Q: What is 123 * 45?
A: <thought> 123 * 45 = (100 + 20 + 3) * (40 + 5) = (100 * 40) + (100 * 5) + (20 * 40) + (20 * 5) + (3 * 40) + (3 * 5) = 4000 + 500 + 800 + 100 + 120 + 15 = 5535 </thought> 5535
```

---

## 2. Subtraction

### ❌ Before (Negative Intermediates)
The original implementation broke subtraction down by magnitude, but if a digit in the subtrahend was larger than the minuend, it generated negative numbers.
```text
Q: What is 321 - 185?
A: <thought> 321 - 185 = 300 - 100 + 20 - 80 + 1 - 5 = 200 + -60 + -4 </thought> 136
```
**Why it failed:** While mathematically correct, language models struggle with negative intermediate numbers because they break standard base-10 digit structures seen in normal text.

### ✅ After (Human-like Borrowing)
We implemented proper borrowing from the next significant digit, exactly as taught in grade school.
```text
Q: What is 321 - 185?
A: <thought> 1 ones - 5 ones (borrow 10) -> 11 - 5 = 6. 2 tens - 8 tens (borrow 100) -> 110 - 80 = 30. 3 hundreds - 1 hundreds -> 300 - 100 = 200. Answer = 200 + 30 + 6 = 236 </thought> 136
```

---

## 3. Division

### ❌ Before (Fake Reasoning)
Like multiplication, the original division trace just stated the operation and dumped the answer.
```text
Q: What is 5535 / 45?
A: <thought> Calculating 5535 divided by 45 </thought> 123
```
**Why it failed:** Division is incredibly hard to compute sequentially. Accuracy was 0.0%.

### ✅ After (Partial Quotients)
We reversed the multiplication process. The model explicitly finds how many hundreds fit, then how many tens fit, and subtracts the remainder at each step.
```text
Q: What is 5535 / 45?
A: <thought> 5535 / 45: 45 * 100 = 4500, remaining = 1035. 45 * 20 = 900, remaining = 135. 45 * 3 = 135, remaining = 0. Answer = 100 + 20 + 3 = 123 </thought> 123
```

---

## 4. Addition

### ✅ Status Quo (Partial Sums)
The addition implementation was actually quite solid from the beginning. It correctly splits the numbers by magnitude and adds them up.
```text
Q: What is 123 + 456?
A: <thought> 123 + 456 = 100 + 400 + 20 + 50 + 3 + 6 = 500 + 70 + 9 </thought> 579
```
This allowed the model to reach 36% accuracy even in the baseline run. We kept this logic unchanged.
