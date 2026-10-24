# RMSNorm: Root Mean Square Layer Normalization

This document provides the mathematical formulation, invariance properties, computational efficiency advantages, and implementation details for **RMSNorm** (Root Mean Square Normalization) used in AIMathLab.

RMSNorm (Zhang & Sennrich, NeurIPS 2019) replaces classical Transformer `LayerNorm` (Ba et al., 2016). It has become the standard normalization layer in modern foundation models such as **LLaMA (1/2/3)**, **Mistral/Mixtral**, **Gemma**, and **Qwen**.

---

## 1. Classical Layer Normalization (LayerNorm)

Given an input token feature vector $x \in \mathbb{R}^d$ (where $d = d_{\text{model}}$ is the hidden dimension), standard LayerNorm enforces zero-mean and unit-variance across the feature dimension:

$$
\mu = \frac{1}{d} \sum_{i=1}^d x_i
$$

$$
\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2 + \epsilon}
$$

$$
y = \text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta
$$

Where:
- $\mu \in \mathbb{R}$ is the mean across the hidden dimension.
- $\sigma \in \mathbb{R}$ is the standard deviation.
- $\epsilon > 0$ is a small numerical stability constant (e.g., $10^{-5}$ or $10^{-6}$).
- $\gamma \in \mathbb{R}^d$ is a learnable scaling parameter (`weight`), initialized to $\mathbf{1}$.
- $\beta \in \mathbb{R}^d$ is a learnable bias parameter (`bias`), initialized to $\mathbf{0}$.
- $\odot$ denotes element-wise vector multiplication.

---

## 2. Mathematical Definition of RMSNorm

Zhang & Sennrich (2019) demonstrated through extensive empirical and theoretical analysis that the computational benefit of LayerNorm is **not** driven by re-centering ($\mu = 0$), but rather by the **scaling property** that stabilizes gradient norms across layers.

By dispensing with the mean computation and mean-centering shift, **RMSNorm** scales the activations strictly by the root mean square of their components:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}
$$

$$
\bar{x}_i = \frac{x_i}{\text{RMS}(x)}
$$

$$
y = \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma
$$

Where:
- $x = (x_1, x_2, \dots, x_d)^T \in \mathbb{R}^d$ is the activation vector of a single token.
- $\text{RMS}(x) \in \mathbb{R}$ is the root mean square scalar statistic of $x$.
- $\gamma \in \mathbb{R}^d$ is the single learnable scaling vector (`self.weight`).
- The learnable bias $\beta$ is removed entirely.

---

## 3. Invariance Properties & Efficiency

### A. Scale Invariance
RMSNorm preserves the crucial **scale-invariance** property of LayerNorm:

$$
\text{RMSNorm}(\alpha x) = \frac{\alpha x}{\sqrt{\frac{1}{d} \sum_{i=1}^d (\alpha x_i)^2 + \epsilon}} \odot \gamma \approx \frac{\alpha x}{\alpha \text{RMS}(x)} \odot \gamma = \text{RMSNorm}(x)
$$

If weights in previous layers grow by a constant factor $\alpha$, the forward activations and gradient updates remain automatically calibrated.

### B. Dropping Shift Invariance
LayerNorm possesses both shift-invariance ($\text{LN}(x + c) = \text{LN}(x)$) and scale-invariance. RMSNorm sacrifices shift-invariance, but Zhang & Sennrich demonstrated that shift-invariance does not contribute to training stability in deep neural networks.

### C. Computational & Memory Speedup
1. **Fewer Reduction Passes:** LayerNorm requires computing two distinct statistical reductions across the hidden dimension: first the mean $\mu$, then the variance $\sigma^2 = \frac{1}{d} \sum (x_i - \mu)^2$. RMSNorm requires only **a single reduction** ($\frac{1}{d} \sum x_i^2$).
2. **Speedup:** Eliminating the first reduction pass and subtraction yields a **10% to 25% speedup** in the normalization kernels.
3. **Parameter Efficiency:** Eliminates the $d$-dimensional $\beta$ bias parameter per layer.

---

## 4. Code References in AIMathLab

In this repository, RMSNorm is implemented in [`model.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model.py) and [`model_kv_cache.py`](file:///Users/sourabhk/Workspace/ExperimentingData/Training/model_kv_cache.py#L320-L335) with fused reciprocal square root (`torch.rsqrt`) for optimal GPU/MPS kernel efficiency:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 1. Compute variance = 1/d * sum(x_i^2)
        # 2. Multiply by reciprocal square root: x / sqrt(variance + eps)
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 3. Element-wise scale by learnable weight
        return self.weight * x_norm
```

`RMSNorm` is registered to normalize inputs prior to self-attention (`ln1`), prior to feed-forward blocks (`ln2`), and at the final output of the transformer backbone before the language modeling head (`ln_f`).

