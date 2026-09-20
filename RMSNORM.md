# RMSNorm: Root Mean Square Normalization

This repository replaces the classic Transformer `LayerNorm` with **RMSNorm**, the modern standard used in state-of-the-art models like LLaMA, Mistral, Gemma, and Qwen.

## What is Layer Normalization?
In deep neural networks, activations can grow wildly out of control or vanish as they pass through dozens of layers. Normalization keeps these activations mathematically stable.

Classic **LayerNorm** stabilizes activations in two steps:
1. **Mean Centering:** It calculates the mean (average) of the activations and subtracts it.
2. **Variance Scaling:** It calculates the variance (spread) of the activations and divides by the square root of the variance, scaling them to a standard deviation of 1.

Finally, it applies a learnable scale (`weight`) and shift (`bias`).

The math looks like this:
$$ \text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \text{weight} + \text{bias} $$

### How Normalization Applies to Tokens
It is a common misconception that these normalization techniques are applied across the entire sentence sequence (the "timestep" dimension). In reality, LayerNorm and RMSNorm operate independently on the **embedding dimension** (the features) for *each individual token*. 

For example, if your sequence is "The cat sat" and your embedding dimension is 384:
- The normalization layer grabs the 384 features for "The" and normalizes them against themselves.
- It grabs the 384 features for "cat" and normalizes them against themselves.
- It grabs the 384 features for "sat" and normalizes them against themselves.

This ensures every individual token remains mathematically stable as it travels deeper into the network's layers, without its representation changing based on the length of the sentence!

## Why switch to RMSNorm?
In 2019, researchers hypothesized that the **mean centering** part of LayerNorm was actually mathematically useless. They proved that the only part of LayerNorm that actually helps the network learn is the **variance scaling**.

Calculating the mean and subtracting it across massive matrices is computationally expensive. **RMSNorm** simply drops the mean-centering step entirely. It also drops the learnable `bias` parameter, keeping only the learnable `weight` scaling parameter.

The math looks like this:
$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\text{RMS}(x)^2 + \epsilon}} \times \text{weight} $$

*(Where RMS is the Root Mean Square, or the square root of the mean of the squared values).*

## Benefits of RMSNorm
1. **Faster Computation:** By dropping the mean subtraction, RMSNorm requires fewer mathematical operations and fewer memory accesses. This results in a **10% to 20% speedup** in the normalization layers.
2. **Identical Performance:** Despite doing less math, models trained with RMSNorm achieve identical (and sometimes slightly better) loss curves and accuracy compared to standard LayerNorm.
3. **Fewer Parameters:** RMSNorm drops the `bias` parameter, making the model slightly more parameter-efficient.

## Implementation Details
In our `RMSNorm` class, the core calculation is extremely compact in PyTorch:
```python
# 1. Square the inputs (x.pow(2))
# 2. Take the mean across the embedding dimension (.mean(-1, keepdim=True))
# 3. Add epsilon and take the reciprocal square root (torch.rsqrt)
x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

# Finally, multiply by the learnable weight parameter
return self.weight * x_norm
```
