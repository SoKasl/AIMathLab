# RoPE: Rotary Position Embeddings

This document provides the mathematical foundations, 2D Givens rotation matrix formulation, relative distance preservation proof, and implementation details for **RoPE (Rotary Position Embeddings)** used in AIMathLab.

RoPE (Su et al., 2021, *RoFormer*) replaces classic absolute positional embedding tables (Vaswani et al., 2017; Radford et al., 2019). It is the standard position representation in **LLaMA (1/2/3)**, **Mistral/Mixtral**, **PaLM**, **Gemma**, and **Qwen**.

---

## 1. The Core Limitation of Absolute Position Embeddings

In classical GPT-2 architectures, position information is injected once at the embedding layer:

$$
x = \text{TokenEmbedding}(w_m) + \text{PositionalEmbedding}(m)
$$

### Critical Failure Modes:
1. **Signal Dissipation:** As $x$ propagates across 6, 12, or 32 layers of non-linear self-attention and feed-forward transformations, the initial additive positional signal degrades.
2. **Failure of Length Generalization:** The model learns a discrete lookup table $\mathbf{P} \in \mathbb{R}^{L_{\text{train}} \times d}$. Any position $m > L_{\text{train}}$ encountered during inference has an uninitialized embedding vector, causing catastrophic collapse on out-of-distribution (OOD) sequence lengths.
3. **No Direct Relative Shift Invariance:** The dot product $(q_m + p_m)^T (k_n + p_n)$ expands into cross-terms $(q_m^T k_n + q_m^T p_n + p_m^T k_n + p_m^T p_n)$ which do not cleanly isolate the relative distance $(n - m)$.

---

## 2. Mathematical Definition of RoPE

RoPE encodes position by **rotating** the Query ($q$) and Key ($k$) vectors in 2D coordinate planes before computing the attention dot product.

Given a Query vector $q_m \in \mathbb{R}^{d_{\text{head}}}$ at sequence index $m$, we pair adjacent components into $\frac{d}{2}$ two-dimensional vectors:

$$
\mathbf{q}_m = \left( [q_m^{(0)}, q_m^{(1)}]^T, [q_m^{(2)}, q_m^{(3)}]^T, \dots, [q_m^{(d-2)}, q_m^{(d-1)}]^T \right)
$$

Each 2D pair $i \in \{0, 1, \dots, \frac{d}{2}-1\}$ is rotated by angle $m \theta_i$:

$$
\begin{pmatrix} \tilde{q}_m^{(2i)} \\ \tilde{q}_m^{(2i+1)} \end{pmatrix} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix} \begin{pmatrix} q_m^{(2i)} \\ q_m^{(2i+1)} \end{pmatrix}
$$

Where the base frequency $\theta_i$ decreases geometrically across channels:

$$
\theta_i = \theta_{\text{base}}^{-2i / d}, \quad \text{typically } \theta_{\text{base}} = 10000.0
$$

In matrix form, the full rotation matrix $\mathbf{R}_{\Theta, m} \in \mathbb{R}^{d \times d}$ is an orthogonal block-diagonal matrix composed of $2 \times 2$ Givens rotation blocks:

$$
\mathbf{R}_{\Theta, m} = \text{diag}\left( \mathbf{R}_0(m\theta_0), \mathbf{R}_1(m\theta_1), \dots, \mathbf{R}_{\frac{d}{2}-1}(m\theta_{\frac{d}{2}-1}) \right)
$$

$$
\tilde{\mathbf{q}}_m = \mathbf{R}_{\Theta, m} \mathbf{q}_m, \qquad \tilde{\mathbf{k}}_n = \mathbf{R}_{\Theta, n} \mathbf{k}_n
$$

---

## 3. Mathematical Proof: Relative Attention Invariance

When computing the self-attention affinity between query $\mathbf{q}$ at position $m$ and key $\mathbf{k}$ at position $n$:

$$
\text{Score}(m, n) = \tilde{\mathbf{q}}_m^T \tilde{\mathbf{k}}_n = (\mathbf{R}_{\Theta, m} \mathbf{q}_m)^T (\mathbf{R}_{\Theta, n} \mathbf{k}_n) = \mathbf{q}_m^T \left( \mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} \right) \mathbf{k}_n
$$

Since $\mathbf{R}_{\Theta, m}$ is orthogonal, its transpose is its inverse, which corresponds to rotation by $-m \theta$:

$$
\mathbf{R}_{\Theta, m}^T = \mathbf{R}_{\Theta, -m}
$$

Because rotation matrices commute under matrix multiplication:

$$
\mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} = \mathbf{R}_{\Theta, -m} \mathbf{R}_{\Theta, n} = \mathbf{R}_{\Theta, n - m}
$$

Substituting back into the attention score:

$$
\tilde{\mathbf{q}}_m^T \tilde{\mathbf{k}}_n = \mathbf{q}_m^T \mathbf{R}_{\Theta, n - m} \mathbf{k}_n = g(\mathbf{q}_m, \mathbf{k}_n, n - m)
$$

### Key Insights:
1. **Pure Relative Distance:** The attention score depends **strictly on the relative displacement $n - m$**, not on absolute positions $m$ or $n$.
2. **Decay with Distance:** As $|n - m|$ increases, higher-frequency components rotate rapidly and cancel out, creating an inductive bias toward local token interactions while preserving long-range capability.

---

## 4. Complex Number Formulation (Implementation Trick)

A 2D rotation of vector $(x, y)$ by angle $\phi$ is identical to multiplying the complex number $z = x + i y$ by the unit phasor $e^{i \phi} = \cos \phi + i \sin \phi$:

$$
\tilde{z} = z \cdot e^{i \phi} = (x \cos \phi - y \sin \phi) + i (x \sin \phi + y \cos \phi)
$$

PyTorch provides native support via `torch.view_as_complex` and `torch.view_as_real`, allowing us to compute all 2D rotations simultaneously as a single element-wise complex multiplication across all batches, heads, and tokens.

---

## 5. RoPE Interaction with KV Caching (`pos_offset`)

A critical requirement when pairing RoPE with autoregressive KV caching:
- During training (or pre-fill), sequence $[0, \dots, T-1]$ is processed at once. Position indices are simply `arange(0, T)`.
- During decoding with KV cache, the model processes **a single new token** ($T_{\text{new}} = 1$) at step $m$.
- If we naively pass `x` (length 1), PyTorch will assign it index $0$, which incorrectly applies rotation angle $0 \cdot \theta$!
- **The Solution:** The generation loop passes `pos_offset = past_tokens_count`. The frequency slice used is:

$$
\text{freqs} = \text{freqs\_cis}[\text{pos\_offset} : \text{pos\_offset} + T]
$$

This guarantees that token $m$ is rotated by $m \theta$, exactly matching the angle it would receive in a non-cached full forward pass.

---

## 6. Code References in AIMathLab

### Precomputing Complex Frequencies & Application: [`model.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model.py) and [`model_kv_cache.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model_kv_cache.py#L38-L74)
```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 1. Compute theta frequencies for each head dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 2. Outer product with position indices [0, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # 3. Form unit complex phasors: e^(i * t * theta) = cos + i*sin
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # Cast real pairs to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # Element-wise complex multiplication (rotation in 2D)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### Rotary Projection in Attention with `pos_offset`: [`model_kv_cache.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model_kv_cache.py#L178-L180)
```python
if self.config.use_rope:
    q, k = apply_rotary_emb(q, k, self.freqs_cis[pos_offset : pos_offset + T])
```

