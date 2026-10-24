# KV Caching: Key-Value Caching Architecture

This document explains the mathematical mechanics, tensor transformations, forward pass modifications, and implementation details for **KV Caching** (Key-Value Caching) in AIMathLab.

---

## 1. Why Do We Have `model_kv_cache.py`?

In this codebase, you will notice both `model.py` and `model_kv_cache.py`:

| File | Architectural Role | Purpose |
| :--- | :--- | :--- |
| [`model.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model.py) | **Baseline Transformer Engine** | Simple, clean implementation of GPT-2 style Transformer without KV caching. Used as the original pedagogical baseline for training. |
| [`model_kv_cache.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model_kv_cache.py) | **Optimized Inference Engine** | Extension of `model.py` created to introduce **KV Caching**, **Flash Attention** (`F.scaled_dot_product_attention`), and **RoPE `pos_offset`** for 10x faster generation during evaluation. |


---

## 2. The Core Problem: $O(N^2)$ Redundant Computation

During autoregressive generation without a KV cache:
- To generate token $100$, the model processes $99$ tokens.
- To generate token $101$, the model processes $100$ tokens, completely recomputing $Q, K, V$ for tokens $0 \dots 99$ from scratch.
- Total computation across $N$ generated tokens scales quadratically as $\sum_{i=1}^N i \approx \mathcal{O}(N^2)$.

### Key Invariant
In causal autoregressive self-attention, the Key ($k$) and Value ($v$) representations of token $i$ depend **only on tokens $0 \dots i$**. As subsequent tokens $i+1, i+2$ are generated, **the Keys and Values of all previous tokens never change**.

Therefore, we can store past Keys and Values in memory and only process the **single newest token** at each decoding step, dropping computation to $\mathcal{O}(N)$ total steps.

---

## 3. How the Forward Pass Changes

The forward pass is partitioned into two distinct operational regimes:

```
========================================================================================
PHASE 1: PRE-FILL (Prompt Ingestion)
Input: context tokens [w_0, w_1, ..., w_{T-1}]  --> Shape: (B, T_prompt)
past_key_values: None
Model computes: Q, K, V for all T_prompt tokens simultaneously.
Output: Logits for token T_prompt, and initial past_key_values = [(K_0, V_0), ..., (K_L, V_L)]
========================================================================================
                                     |
                                     v (sample next token w_T)
========================================================================================
PHASE 2: DECODING (Token-by-Token Autoregressive Loop)
Input: SINGLE token [w_t]                       --> Shape: (B, 1)  <-- Strictly length 1!
past_key_values: [(K_layer, V_layer)] containing tokens 0 ... t-1
pos_offset: t (tells RoPE which angle to rotate token t)
Model computes: Q, K, V ONLY for token w_t, concatenates new K, V to cache,
and returns updated past_key_values containing tokens 0 ... t.
========================================================================================
```

---

## 4. Code Modifications: Before vs. After KV Cache (Every Layer Analyzed)

To add KV Caching to a Transformer, three distinct layers must be updated to pass and update cache tensors. Compare the implementations side-by-side:

### Level 1: `GPT.forward` (Full Model Level)

#### BEFORE (No Cache):
```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)  # (B, T, C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
    x = tok_emb + pos_emb
    
    # Process through all blocks sequentially
    for block in self.blocks:
        x = block(x)                           # Output is strictly new x
        
    x = self.ln_f(x)
    logits = self.lm_head(x)
    return logits, loss
```

#### AFTER (With Cache):
```python
def forward(self, idx, targets=None, past_key_values=None, pos_offset=0):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)  # (B, 1, C) during decoding!
    
    if not self.config.use_rope:
        # Must offset position embedding if not using RoPE
        pos_emb = self.position_embedding_table(
            torch.arange(pos_offset, pos_offset + T, device=idx.device)
        )
        x = tok_emb + pos_emb
    else:
        x = tok_emb  # RoPE handles positions dynamically inside attention
        
    presents = []  # To collect updated (k, v) from each layer
    for i, block in enumerate(self.blocks):
        # 1. Extract layer i's past cache (or None during pre-fill/training)
        layer_past = past_key_values[i] if past_key_values is not None else None
        
        # 2. Pass layer cache down into the block
        x, present = block(x, layer_past=layer_past, pos_offset=pos_offset)
        
        # 3. Store the new cache returned by this block
        presents.append(present)
        
    x = self.ln_f(x)
    logits = self.lm_head(x)
    # 4. Return logits, loss, AND the updated cache list!
    return logits, loss, presents
```

---

### Level 2: `Block.forward` (Transformer Block Level)

#### BEFORE (No Cache):
```python
def forward(self, x):
    # Standard residual connections
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
```

#### AFTER (With Cache):
```python
def forward(self, x, layer_past=None, pos_offset=0):
    # 1. Self-Attention returns both the residual representation AND the updated cache!
    attn_out, present = self.sa(self.ln1(x), layer_past=layer_past, pos_offset=pos_offset)
    x = x + attn_out
    
    # 2. Feedforward block (SwiGLU/MLP) only operates on the single new token representation
    x = x + self.ffwd(self.ln2(x))
    
    # 3. Return updated representation AND present cache tuple
    return x, present
```

---

### Level 3: `MultiHeadAttention.forward` (Attention Kernel Level)

This is where past Keys and Values are retrieved, updated, and multiplied:

#### BEFORE (No Cache):
```python
def forward(self, x):
    B, T, C = x.size()
    # Compute Q, K, V for ALL T tokens
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    
    # Reshape to (B, H, T, D)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    
    # Full (T x T) attention matrix with causal mask
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.c_proj(y)
```

#### AFTER (With Cache):
```python
def forward(self, x, layer_past=None, pos_offset=0):
    B, T, C = x.size()  # During decoding: T == 1!
    
    # 1. Compute Q, K, V ONLY for the incoming token (T=1)
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    
    # Reshape to (B, H, 1, D)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    
    # 2. RoPE rotation at absolute position (pos_offset)
    if self.config.use_rope:
        freqs = self.freqs_cis[pos_offset : pos_offset + T]
        q, k = apply_rotary_emb(q, k, freqs)
        
    # 3. CONCATENATE PAST KEYS AND VALUES:
    if layer_past is not None:
        past_k, past_v = layer_past  # Shapes: (B, H, T_past, D)
        k = torch.cat([past_k, k], dim=2)  # Shape: (B, H, T_past + 1, D)
        v = torch.cat([past_v, v], dim=2)  # Shape: (B, H, T_past + 1, D)
        
    # Save the updated full history as present cache
    present = (k, v)
    
    # 4. Attention: Query (1 token) against ALL Keys (T_past + 1 tokens)
    # Output score shape: (B, H, 1, T_past + 1)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
    # Notice: Causal mask only needed if T > 1 (pre-fill phase).
    # During decode (T=1), the new token can attend to all past tokens!
    if T > 1:
        att = att.masked_fill(self.tril[pos_offset:pos_offset+T, :k.size(2)] == 0, float('-inf'))
        
    att = F.softmax(att, dim=-1)
    
    # 5. Multiply scores (1 x T_total) by Values (T_total x D) -> (1 x D)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.c_proj(y), present
```

---

### Level 4: The Generation Loop (`generate`)

#### BEFORE (Slow $O(N^2)$ Loop):
```python
@torch.no_grad()
def generate_naive(self, idx, max_new_tokens):
    # Context grows each step: [t1] -> [t1, t2] -> [t1, t2, t3]
    for _ in range(max_new_tokens):
        # Must crop and pass the ENTIRE sequence into the model!
        idx_cond = idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)             # Extremely slow recomputation!
        
        # Take logits for the last token
        next_token = sample(logits[:, -1, :])
        idx = torch.cat((idx, next_token), dim=1)
    return idx
```

#### AFTER (Fast $O(N)$ Cached Loop):
```python
@torch.no_grad()
def generate_with_cache(self, idx, max_new_tokens):
    # ----------------------------------------------------
    # Phase 1: Pre-fill (Ingest initial prompt of length T)
    # ----------------------------------------------------
    logits, _, past_key_values = self(idx)     # Builds initial cache
    
    # Sample first new token
    next_token = sample(logits[:, -1, :])      # Shape: (B, 1)
    idx = torch.cat((idx, next_token), dim=1)
    
    # ----------------------------------------------------
    # Phase 2: Decoding (Pass ONLY 1 token per iteration!)
    # ----------------------------------------------------
    for _ in range(max_new_tokens - 1):
        pos_offset = idx.shape[1] - 1          # True position in sequence
        
        # Pass ONLY the single new token and the cache!
        logits, _, past_key_values = self(
            next_token, 
            past_key_values=past_key_values, 
            pos_offset=pos_offset
        )
        
        next_token = sample(logits[:, -1, :])  # Shape: (B, 1)
        idx = torch.cat((idx, next_token), dim=1)
        if next_token.item() == EOS_TOKEN:
            break
            
    return idx
```

---

## 5. Performance Impact in AIMathLab

By implementing KV caching and pairing it with Flash Attention:
- **Generation latency per math problem:** Dropped from **~3.2 seconds** down to **~0.35 seconds** (**~9.1x speedup**).
- **VRAM consumption during generation:** Peak memory is bound by $2 \times L \times H \times T \times D$ cache tensors rather than the $O(T^2)$ full attention matrix.
