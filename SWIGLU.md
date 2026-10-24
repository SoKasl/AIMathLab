# SwiGLU: Swish-Gated Linear Unit

This document provides the mathematical foundations, architectural derivation, parameter parity scaling, and implementation details for the **SwiGLU** activation network used in AIMathLab.

SwiGLU replaces the classic 2-layer FeedForward Network (FFN) found in original Transformer architectures (such as Vaswani et al., 2017 and GPT-2) with a 3-matrix gated bilinear network. It is now the standard FFN in modern foundation models including **LLaMA (1/2/3)**, **Mistral/Mixtral**, **PaLM**, and **Gemma**.

---

## 1. Classical Transformer MLP (Baseline)

In a standard Transformer block, the output of the multi-head self-attention layer is normalized and passed through a 2-layer Multi-Layer Perceptron (MLP):

$$
\text{FFN}_{\text{standard}}(x) = \text{Activation}(x W_1 + b_1) W_2 + b_2
$$

Where:
- $x \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ is the input token representation.
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$ is the expansion (up-projection) weight matrix, typically with $d_{\text{ffn}} = 4 \times d_{\text{model}}$.
- $W_2 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$ is the contraction (down-projection) weight matrix.
- $\text{Activation}(\cdot)$ is historically **ReLU** ($f(z) = \max(0, z)$) in original Transformers, or **GELU** ($f(z) = z \cdot \Phi(z)$) in GPT-2/3.

In this classical formulation, every feature channel is transformed independently and monotonically by a single static activation curve.

---

## 2. Mathematical Definition of GLU (Gated Linear Unit)

Introduced by Dauphin et al. (2017), the **Gated Linear Unit (GLU)** introduces a multiplicative (bilinear) branch where one linear projection dynamically modulates or "gates" the information flowing through a second linear projection:

$$
\text{GLU}(x, W, V, b, c) = \sigma(x W + b) \odot (x V + c)
$$

Where:
- $x W + b$ is the **Gate Projection**.
- $\sigma(\cdot)$ is the component-wise sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$, acting as a soft continuous switch in the range $(0, 1)$.
- $x V + c$ is the **Linear Feature Projection** (unbounded).
- $\odot$ denotes the element-wise (Hadamard) vector product.

### Why Gating Matters for Mathematical Reasoning
Arithmetic algorithms require sharp, context-dependent routing decisions:
- If `carry == 1`, route activation to increment column $k+1$.
- If `digit_sum >= 10`, emit `carry = 1` and subtract `10`.

A standard MLP must approximate these hard step functions using linear combinations of static sigmoids or ReLUs. In contrast, a Gated Linear Unit provides a **differentiable multiplicative switch** at every single hidden dimension, allowing the model to zero out irrelevant features or amplify active branches conditionally on the input token.

---

## 3. The SwiGLU Variant (Shazeer, 2020)

In *"GLU Variants Improve Transformer"* (Noam Shazeer, Google, 2020), various activation functions were benchmarked to replace the standard sigmoid $\sigma(\cdot)$ in the gate projection:

| Variant | Gating Function Formulation |
| :--- | :--- |
| **ReGLU** | $\text{ReGLU}(x) = \max(0, x W) \odot (x V)$ |
| **GEGLU** | $\text{GEGLU}(x) = \text{GELU}(x W) \odot (x V)$ |
| **SwiGLU** | $\text{SwiGLU}(x) = \text{Swish}_1(x W) \odot (x V)$ |

---

## 4. Difference Between Sigmoid and Swish (SiLU)

The mathematical relationship between **Sigmoid** and **Swish** is direct:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

**Swish is literally $z$ multiplied by its own Sigmoid!**

### Comparative Analysis:

| Property | Standard Sigmoid: $\sigma(z)$ | Swish / SiLU: $z \cdot \sigma(z)$ | Mathematical Impact |
| :--- | :--- | :--- | :--- |
| **Output Bounds** | Strictly bounded in $(0, 1)$ | Lower-bounded at $\approx -0.28$, unbounded above $[ -0.28, +\infty )$ | Swish can scale feature magnitude dynamically without artificial ceilings. |
| **Large Positive Inputs ($z \gg 0$)** | **Saturates at $1.0$** (Derivative $\to 0$) | **Acts like $z$** ($z \cdot 1.0 = z$, Derivative $\to 1.0$) | **Eliminates vanishing gradients**, enabling deep Transformers to backpropagate freely. |
| **Value at $z = 0$** | $\sigma(0) = 0.5$ | $\text{Swish}(0) = 0.0$ | Naturally zero-centered at origin. |
| **Negative Inputs ($z < 0$)** | Asymptotically approaches $0.0$ | Smooth negative dip (minimum $\approx -0.278$ at $z \approx -1.28$) | Preserves small negative gradients; avoids the "Dying ReLU" neuron death problem. |

### Full SwiGLU FFN Equation

Dropping biases (as is standard in modern LLMs to improve training stability and throughput), the complete SwiGLU feed-forward network with down-projection $W_2$ is defined as:

$$
\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{SiLU}(x W_1) \odot (x W_3) \right) W_2
$$

Where:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$: Gate projection matrix
- $W_3 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$: Up projection matrix
- $W_2 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$: Down projection matrix
- $\odot$: Element-wise tensor multiplication

---

## 5. Parameter Parity Math ($8/3 \times d_{\text{model}}$ Scaling)

A standard FFN has **2 weight matrices**:

$$
\text{Params}_{\text{standard}} = (d_{\text{model}} \times 4 d_{\text{model}}) + (4 d_{\text{model}} \times d_{\text{model}}) = 8 d_{\text{model}}^2
$$

Because SwiGLU introduces a 3rd matrix ($W_3$), setting $d_{\text{ffn}} = 4 d_{\text{model}}$ would increase the parameter count by $50\%$ to $12 d_{\text{model}}^2$.

To maintain **exact parameter parity** with a standard 2-layer MLP, the hidden dimension $d_{\text{ffn}}$ is scaled down by a factor of $\frac{2}{3}$:

$$
d_{\text{ffn}} = \left\lfloor \frac{2}{3} \times (4 d_{\text{model}}) \right\rfloor = \left\lfloor \frac{8}{3} d_{\text{model}} \right\rfloor
$$

With this scaling:

$$
\text{Params}_{\text{SwiGLU}} = 3 \times \left( d_{\text{model}} \times \frac{8}{3} d_{\text{model}} \right) = 8 d_{\text{model}}^2
$$

In AIMathLab ($d_{\text{model}} = 384$):
- Standard MLP: $d_{\text{ffn}} = 4 \times 384 = 1536 \implies 2 \times (384 \times 1536) = 1,179,648\text{ parameters}$
- SwiGLU: $d_{\text{ffn}} = \lfloor \frac{8}{3} \times 384 \rfloor = 1024 \implies 3 \times (384 \times 1024) = 1,179,648\text{ parameters}$

The parameter counts are identical to the single byte.

---

## 6. Before vs. After Code Comparison: Standard MLP vs. SwiGLU

### BEFORE: Standard 2-Layer MLP (e.g., Classical GPT-2)
```python
class StandardMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Expand from d_model (384) to 4 * d_model (1536)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        # 2. Project back down from 1536 to 384
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 384)
        h = self.c_fc(x)           # Shape: (Batch, Seq_Len, 1536)
        h = self.gelu(h)           # Shape: (Batch, Seq_Len, 1536)
        out = self.c_proj(h)       # Shape: (Batch, Seq_Len, 384)
        return self.dropout(out)
```

### AFTER: Modern 3-Matrix SwiGLU (LLaMA / Gemma / AIMathLab)
```python
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Hidden dimension scaled to 8/3 * d_model (1024) to keep total parameters identical!
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        
        # 3 Linear Projections instead of 2:
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # 1. GATE projection
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # 2. UP projection
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)  # 3. DOWN projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 384)
        
        # Branch 1: The Gate (determines what information to let through)
        gate = F.silu(self.w1(x))   # Shape: (Batch, Seq_Len, 1024)
        
        # Branch 2: The Signal (unbounded feature representation)
        up = self.w3(x)             # Shape: (Batch, Seq_Len, 1024)
        
        # Branch 3: Element-wise Hadamard Product (Bilinear Gating)
        gated_signal = gate * up    # Shape: (Batch, Seq_Len, 1024)
        
        # Down-projection back to embedding dimension
        out = self.w2(gated_signal) # Shape: (Batch, Seq_Len, 384)
        return self.dropout(out)
```

---

### Step-by-Step Tensor Dimension Trace (Concrete Example: $B=2, T=5, d=384$)

```
Input Tensor: x ∈ (2, 5, 384)
 │
 ├──► self.w1(x)  ∈ (2, 5, 1024)  ──► F.silu(...)  ──► gate  ∈ (2, 5, 1024)
 │                                                         │
 ├──► self.w3(x)  ∈ (2, 5, 1024)  ─────────────────────► up    ∈ (2, 5, 1024)
 │                                                         │
 │                                                         ▼
 │                                             (gate * up) ∈ (2, 5, 1024)
 │                                                         │
 └─────────────────────────────────────────────────────────┼──► self.w2(...) ──► out ∈ (2, 5, 384)
```

## 7. Code References in AIMathLab

In this repository, SwiGLU is implemented in [`model.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model.py) and [`model_kv_cache.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model_kv_cache.py#L285-L305):

```python
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Industry standard: scale by 8/3 to keep parameters identical to a 4x standard MLP
        hidden_dim = int(8 * config.n_embd / 3)
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # Gate projection
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # Up projection
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)  # Down projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU(x, W1, W2, W3) = W2 * ( Swish(W1 * x) ⊗ (W3 * x) )
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.dropout(self.w2(gate * up))
```

This module is used both as the primary FeedForward block inside `Block` and as the expert network inside `SparseMoE`.

