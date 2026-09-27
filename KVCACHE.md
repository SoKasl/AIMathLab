# KV Caching

KV Caching (Key-Value Caching) is a crucial optimization for running autoregressive transformer models (like GPT) efficiently during inference (generation).

## The Problem: $O(N^2)$ Generation
During training, we use "Teacher Forcing." We pass the entire sequence into the transformer at once, and it computes the attention for all tokens in parallel.

However, during inference, we generate text one token at a time. Without KV Caching, to generate the 100th token, the model has to process a sequence of 99 tokens. To generate the 101st token, it processes a sequence of 100 tokens. 
At every step, it re-computes the Query, Key, and Value vectors for **all previous tokens** from scratch. 
This means generation becomes increasingly slow as the sequence grows, resulting in $O(N^2)$ time complexity.

> [!NOTE]
> **Training vs Inference Speed**
> KV Caching **does not speed up training** (the backward pass and gradient updates). During training, we use Teacher Forcing where the entire sequence is known, so the model computes attention for all tokens simultaneously in parallel. There is no step-by-step loop to cache. 
> 
> However, KV Caching **drastically speeds up our experiment script**. Why? Because during training, our script frequently pauses to evaluate the model's accuracy on math problems. During these evaluation phases, the model must *generate* the answers token-by-token. By dropping generation time from ~3.2s to ~0.35s per problem, our overall training script finishes much faster!

## The Solution: KV Caching
In self-attention, the output for the current token depends on its own Query vector, and the Key and Value vectors of all previous tokens.
Crucially, the Key and Value vectors of previous tokens **do not change** as new tokens are added.

Therefore, we can simply save (cache) the Keys and Values in memory!
When generating the next token:
1. We only pass the **single newest token** through the model.
2. The model computes the Q, K, V for just this one token.
3. We append the new K and V to our saved cache.
4. We compute attention using the new Q and the *entire* cached K and V.

This drops the generation time complexity to $O(N)$ per token.

## Educational Code Example
Here is the conceptual logic of KV Caching in PyTorch:

```python
# 1. We only pass the new token through the linear layers
qkv = self.c_attn(new_token)
q, k, v = qkv.split(self.n_embd, dim=2)

# 2. Append the new K and V to the cache
if layer_past is not None:
    past_k, past_v = layer_past
    k = torch.cat((past_k, k), dim=-2)  # append along sequence dimension
    v = torch.cat((past_v, v), dim=-2)

present = (k, v) # Save for the next token

# 3. Compute Attention
# Notice we don't need a causal mask because we are only querying for the last token!
attention_scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
attention_weights = F.softmax(attention_scores, dim=-1)
out = attention_weights @ v
out = attention_weights @ v
```

## How `layer_past` is passed: The Generation Loop

To understand where `layer_past` comes from, we have to look at the generation loop that calls the model.

### 1. Original Code (Without KV Cache)
In the original autoregressive loop, we pass the **entire growing sequence** into the model every single time. The model computes everything from scratch, and we just grab the last token's output. We don't save anything between steps.

```python
# The context sequence keeps growing: [A] -> [A, B] -> [A, B, C]
for _ in range(max_new_tokens):
    # Pass the ENTIRE sequence into the model (Slow!)
    logits, _ = model(context)
    
    # We only care about the last token's prediction
    next_token_logits = logits[:, -1, :]
    
    # Sample and append it to the context
    next_token = sample(next_token_logits)
    context = torch.cat((context, next_token), dim=1)
```

### 2. New Code (With KV Cache)
With KV Caching, the process is split into two distinct phases: **Pre-filling** and **Decoding**. The model returns a new object: `past_key_values`. This is a list containing the `present = (k, v)` tuples from every transformer block. In the next step of the loop, we pass those *back* into the model as `layer_past`.

```python
# ==========================================
# PHASE 1: PRE-FILL
# ==========================================
# We pass the initial context (e.g., 10 tokens) into the model.
# The model computes Q, K, V for all 10 tokens.
# It returns the logits, and saves the 10-token K and V into `past_key_values`.
logits, _, past_key_values = model(context)

# We grab the prediction for the 11th token
next_token = sample(logits[:, -1, :])
context = torch.cat((context, next_token), dim=1)

# ==========================================
# PHASE 2: DECODING (Token-by-Token)
# ==========================================
for _ in range(max_new_tokens - 1):
    
    # 1. Pass ONLY the 11th token (length 1) into the model.
    #    We also pass in the cache containing tokens 1-10.
    logits, _, past_key_values = model(next_token, past_key_values=past_key_values)
    
    # 2. Inside the model:
    #    - The model computes Q, K, V for just the 11th token.
    #    - It appends the 11th token's K and V to the cache (now size 11).
    #    - It computes attention using the 11th token's Q against all 11 K/V pairs!
    #    - It returns the updated cache.
    
    # 3. We get the prediction for the 12th token.
    next_token = sample(logits[:, -1, :])
    context = torch.cat((context, next_token), dim=1)
```

**Why is this so fast?**
In the old code, generating the 100th token required running linear layers and attention on all 99 previous tokens again. 
In the new code, generating the 100th token requires running linear layers on exactly **1 token**, and doing a simple attention dot product against the cached memory.
When using Rotary Position Embeddings (RoPE) with KV Caching, we must be careful.
Because we only pass a sequence of length 1 (the new token) through the model, the model loses track of its absolute position in the sequence. 
If we generate the 100th token, we must explicitly pass `pos_offset = 99` so that RoPE applies the 100th rotation angle to the new token's Query and Key vectors.

## Where is the code?
To keep the primary `model.py` extremely clean for learning, the KV Cache implementation is provided as a separate, fully-featured subclass in [`model_kv_cache.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model_kv_cache.py).

If you look at that file, you will find:
1. `_forward_flash` accepts a `layer_past` caching mechanism.
2. `apply_rotary_emb` accepts a `pos_offset` so it rotates by absolute sequence position rather than `[0:1]`.
3. `GPT.generate_with_cache` is a highly-optimized generation loop that pre-fills the cache and then runs exactly 1 token at a time.
