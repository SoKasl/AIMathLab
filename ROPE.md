# RoPE: Rotary Position Embeddings

This repository replaces classic Absolute Position Embeddings with **RoPE (Rotary Position Embeddings)**, which is the gold standard for positioning in modern LLMs like LLaMA, Mistral, and GPT-4.

## The Flaw of Absolute Position Embeddings
In classic Transformers (like GPT-2), positional information is added at the very beginning of the network:
```python
x = token_embedding + absolute_position_embedding
```
**The Problem:** As the tensor `x` passes through 10, 20, or 100 layers of complex attention and feed-forward transformations, the original positional signal gets "washed out". By layer 50, the model has a very blurry understanding of exactly how far apart two words are. 

Worse, the model struggles to generalize to sequences longer than it was trained on because it only memorized *absolute* positions (e.g., "Word #500"), rather than understanding *relative* distances (e.g., "Word A is 3 words away from Word B").

## The RoPE Solution
RoPE solves this by completely dropping the absolute position embeddings at the start. Instead, it injects positional information **directly inside the Attention mechanism at every single layer**.

### Every Layer vs Initial Injection
This is a crucial distinction!
- **Absolute Embeddings:** Only happen **once**, at layer 0, before the Transformer blocks even start. The network has to "remember" this initial signal across all layers.
- **RoPE:** Happens **inside every single Transformer block**. If your model has 12 layers, the RoPE rotation is applied 12 times. 

It does this dynamically: right after a Transformer layer computes the raw Query (`Q`) and Key (`K`) vectors, RoPE intercepts them and mathematically **rotates** them based on their positions before allowing them to compute their attention scores.

### How Rotation captures Relative Distance
Imagine a clock face. 
- Token A is at the 12 o'clock position (0 degrees).
- Token B is at the 1 o'clock position (30 degrees).
- The relative distance between them is 30 degrees.

If we shift both tokens forward by 5 positions:
- Token A is now at the 5 o'clock position (150 degrees).
- Token B is now at the 6 o'clock position (180 degrees).
- **The relative distance between them is STILL 30 degrees.**

Because the Attention mechanism uses a **dot product** (which calculates the angle between two vectors), rotating the Q and K vectors based on their positions mathematically guarantees that the attention score between two words depends **only on their relative distance**, no matter where they appear in the sentence!

## Implementation Details
1. **Bypass Absolute Embeddings:** In `GPT.forward()`, if `config.use_rope` is active, we completely skip adding the `position_embedding_table` to the tokens.
2. **Complex Numbers:** We convert the `Q` and `K` vectors into complex numbers (`torch.view_as_complex`) because multiplying complex numbers is the fastest and most elegant way to compute 2D rotation.
3. **Layer-by-Layer Injection:** At the start of every single `MultiHeadAttention` block, the newly generated `Q` and `K` vectors are rotated by multiplying them by pre-computed frequency vectors (`freqs_cis`).

### The Code Breakdown
The mathematical rotation is handled by 3 helper functions in `model.py`:

1. `precompute_freqs_cis(dim, end)`
   - **What it does:** Generates a lookup table of rotation angles for every possible position up to the maximum sequence length (`end`).
   - **How it works:** It uses Euler's formula to map positions to angles on a circle, returning them as a tensor of complex numbers (polar coordinates). We compute this once and register it as a permanent buffer (`self.freqs_cis`) so we don't waste GPU cycles recomputing angles every single step.

2. `reshape_for_broadcast(freqs_cis, x)`
   - **What it does:** A simple utility to reshape the 1D list of angles so it matches the 4D shape of the Query/Key tensors `(Batch, Heads, Sequence Length, Head Size)`. 
   - **How it works:** This ensures PyTorch can broadcast the math efficiently across all heads and batches simultaneously.

3. `apply_rotary_emb(xq, xk, freqs_cis)`
   - **What it does:** The actual rotation engine. It intercepts the raw `Q` and `K` vectors and rotates them.
   - **How it works:** 
     1. It groups the real numbers in the Q and K vectors into pairs (x, y coordinates).
     2. Casts them into Complex numbers (`torch.view_as_complex`).
     3. Multiplies the complex Q/K numbers by the complex `freqs_cis` frequencies. In complex math, multiplying two numbers adds their angles together—meaning **the vectors are physically rotated on a 2D plane**. *(For a fantastic visual animation of exactly how this 2D rotation works, check out [EleutherAI's Visual Guide to Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/))*
     4. Finally, it casts them back to real numbers (`torch.view_as_real`) and flattens them back into their original shape to compute Attention.
