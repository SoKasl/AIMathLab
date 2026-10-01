# Hierarchical Developmental Curriculum & Digit-by-Digit Long Division

## 1. Overview & Theoretical Motivation

Standard autoregressive language models, particularly compact "Tiny" architectures (~10M parameters), struggle with complex multi-step arithmetic operations such as multi-digit multiplication and division. Prior work treats arithmetic as flat multi-task fine-tuning, training models on addition, subtraction, multiplication, and division simultaneously.

However, arithmetic is inherently **hierarchically dependent**:
1. **Multi-digit Addition and Subtraction** form the basis for column operations and carry/borrow state management.
2. **Single-digit Multiplication ($0 \times 0$ to $9 \times 9$)** is a primitive memory retrieval task (100 distinct facts) required for all higher operations.
3. **Multi-digit Multiplication** is a composition of single-digit multiplication, base-10 zero-padding, and multi-digit addition (FOIL algorithm).
4. **Long Division** is the most complex algorithmic task, requiring iterative quotient digit estimation, single-digit multiplication, remainder subtraction, and digit descent ("bringing down the next digit").

If a model has not mastered single-digit multiplication and subtraction, long division collapses at every step. This document defines a **4-Stage Developmental Curriculum** and an **Explicit Digit-by-Digit Long Division Scratchpad**.

---

## 2. Curriculum Staging & Data Distribution

We structure the training into four distinct progressive stages. Each stage preserves a rehearsal proportion of earlier operations to prevent **catastrophic forgetting**.

### Stage 1: Additive Foundations
* **Target:** Master place-value decomposition, carry-over, and borrow subtraction.
* **Composition:**
  * 50% Multi-digit Addition (2-digit + 2-digit, 3-digit + 3-digit)
  * 50% Multi-digit Subtraction (with borrow state tracking)
* **Target Size:** 30,000 examples

### Stage 2: Single-Digit Multiplication & Primitive Recall
* **Target:** Memorize the complete $10 \times 10$ multiplication table ($0 \times 0$ through $9 \times 9$) with 100% coverage, while maintaining addition/subtraction.
* **Composition:**
  * 30% Multi-digit Addition
  * 30% Multi-digit Subtraction
  * 40% 1-Digit Multiplication (exhaustive coverage of all 100 pairs, heavily repeated with varied scratchpad representations)
* **Target Size:** 30,000 examples

### Stage 3: Multi-Digit Multiplication with Zero-Padding
* **Target:** Master FOIL expansion and base-10 zero accounting ($2\times 1$, $2\times 2$, $3\times 2$, $3\times 3$).
* **Composition:**
  * 20% Multi-digit Addition
  * 20% Multi-digit Subtraction
  * 60% Multi-digit Multiplication with Explicit Zero-Padding Curriculum
* **Target Size:** 40,000 examples

### Stage 4: Long Division Integration (Full-Stack Mastery)
* **Target:** Master digit-by-digit long division algorithm using the primitive skills acquired in Stages 1–3.
* **Composition:**
  * 10% Addition
  * 10% Subtraction
  * 35% Multi-digit Multiplication
  * 45% Digit-by-Digit Long Division (1-digit and 2-digit divisors)
* **Target Size:** 50,000 examples

---

## 3. Scratchpad Syntax Standards

To ensure token efficiency within the 256-token context window and clean parsing during evaluation, all operations adhere to strict Chain-of-Thought formats.

### 3.1 Addition
```text
Q: What is 476 + 697?
A: <thought> 476 + 697 = 400 + 600 + 70 + 90 + 6 + 7 = 1000 + 160 + 13 = 1173 </thought> 1173<|endoftext|>
```

### 3.2 Subtraction (Borrowing)
```text
Q: What is 801 - 500?
A: <thought> 1 ones - 0 ones -> 1 - 0 = 1. 0 tens - 0 tens -> 0 - 0 = 0. 8 hundreds - 5 hundreds -> 800 - 500 = 300. Answer = 300 + 0 + 1 = 301 </thought> 301<|endoftext|>
```

### 3.3 Multiplication (Zero-Padding Curriculum)
```text
Q: What is 80 * 50?
A: <thought> 8 * 5 = 40. Count zeros: 80 has 1 zero, 50 has 1 zero. Total zeros = 2. Append 2 zeros to 40 -> 4000. </thought> 4000<|endoftext|>
```

### 3.4 Digit-by-Digit Long Division (Standard Algorithm)
For long division, each step must explicitly show:
1. Current working dividend (formed by remainder from prior step concatenated with current digit).
2. Quotient digit determination ($d \times q \le \text{current}$).
3. Product calculation and subtraction to obtain new remainder.
4. Bringing down the next digit until dividend is exhausted.

**Example 1: Single-digit Divisor ($785 \div 5 = 157$)**
```text
Q: What is 785 / 5?
A: <thought>
Divide 785 by 5:
- Digit 7: 7 / 5 = 1 (5 * 1 = 5), rem = 7 - 5 = 2. Quotient digit: 1.
- Bring down 8 -> 28: 28 / 5 = 5 (5 * 5 = 25), rem = 28 - 25 = 3. Quotient digit: 5.
- Bring down 5 -> 35: 35 / 5 = 7 (5 * 7 = 35), rem = 35 - 35 = 0. Quotient digit: 7.
Final quotient: 157.
</thought> 157<|endoftext|>
```

**Example 2: Two-digit Divisor ($321090 \div 973 = 330$)**
```text
Q: What is 1568 / 14?
A: <thought>
Divide 1568 by 14:
- Take 15: 15 / 14 = 1 (14 * 1 = 14), rem = 15 - 14 = 1. Quotient digit: 1.
- Bring down 6 -> 16: 16 / 14 = 1 (14 * 1 = 14), rem = 16 - 14 = 2. Quotient digit: 1.
- Bring down 8 -> 28: 28 / 14 = 2 (14 * 2 = 28), rem = 28 - 28 = 0. Quotient digit: 2.
Final quotient: 112.
</thought> 112<|endoftext|>
```

---

## 4. Evaluation Benchmark
The evaluation set consists of 100 balanced problems (25 per operation: Addition, Subtraction, Multi-digit Multiplication, and Long Division) generated using the exact same digit-by-digit and curriculum formats, saved in `math_eval_hierarchical.txt`.

---

## 5. Unified Dataset Architecture & Training Protocol

Rather than orchestrating four disjoint training scripts with manual checkpoint reloading, all 4 developmental stages are concatenated sequentially into a single unified training corpus:

* **File:** `math_hierarchical_curriculum_train.txt`
* **Total Samples:** 150,000 complete problems
* **Structure:**
  * **Tokens 0 – 20% (Samples 1–30,000):** Stage 1 (Additive Foundations)
  * **Tokens 20% – 40% (Samples 30,001–60,000):** Stage 2 (Single-digit Multiplication Lookup + Add/Sub Rehearsal)
  * **Tokens 40% – 67% (Samples 60,001–100,000):** Stage 3 (Multi-digit FOIL Expansion + Add/Sub Rehearsal)
  * **Tokens 67% – 100% (Samples 100,001–150,000):** Stage 4 (Digit-by-Digit Long Division + Full-Stack Rehearsal)

### 5.1 Benefits of Single-Epoch Sequential Streaming
1. **Continuous Sequence Packing:** Works seamlessly across the entire 150,000-sample stream with ~97% active token density and zero discarded boundaries.
2. **Smooth Learning Rate Annealing:** A unified cosine learning rate schedule warms up during Stage 1 and gradually decays as the network reaches the hardest operations (multi-digit FOIL and long division).
3. **Continuous Rehearsal (Anti-Forgetting):** Because each stage retains a calibrated fraction of earlier operations, the model continually reinforces additive and single-digit primitives while incorporating new algorithmic mechanisms.

### 5.2 Dynamic Evolution Tracking via Periodic Evaluation
During the single epoch through `math_hierarchical_curriculum_train.txt`, the model is evaluated periodically (e.g., every 500 steps) on `math_eval_hierarchical.txt`:
* **Phase 1 (Early Steps):** Addition & Subtraction curves rise; Multiplication and Division remain near 0%.
* **Phase 2 (Mid Steps):** Single-digit multiplication memorization unlocks multi-digit FOIL execution; Multiplication curve climbs.
* **Phase 3 (Late Steps):** With multiplication and subtraction primitives mastered, digit descent in Long Division stabilizes, unlocking the division curve.
