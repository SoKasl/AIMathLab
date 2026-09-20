# SwiGLU: Swish-Gated Linear Unit

This repository implements the **SwiGLU** activation network, a massive architectural upgrade that replaces the standard FeedForward block in classic Transformer models.

## Standard Activations (ReLU & GELU)
Before understanding GLU, it helps to understand standard activations. Historically, the most common activation function was **ReLU** (Rectified Linear Unit), which simply outputs `x` if `x > 0`, and `0` otherwise. It acts as a basic threshold.
Modern classic MLPs (like the original GPT-2) upgraded from ReLU to **GELU** (Gaussian Error Linear Unit), which is a smoother curve but serves the same purpose.

In a classic MLP, the output of the attention layer is passed through two linear layers with that single activation function in between:
```
Classic = Linear2( Activation( Linear1(x) ) )
```
Here, the activation function (like ReLU) applies universally to the single stream of data.

## What is a GLU (Gated Linear Unit)?

A **Gated Linear Unit** splits the first linear layer into two separate pathways:
1. **The Gate:** Transforms the input and passes it through an activation function (determining "how much" of the signal should pass).
2. **The Up-Projection:** Transforms the input linearly without an activation.

These two pathways are then multiplied together element-wise (the mathematical symbol $\otimes$), and finally passed through a down-projection layer:
```
GLU = Linear_down( Activation(Linear_gate(x)) ⊗ Linear_up(x) )
```

## Why Gating Matters for Math
Arithmetic relies heavily on conditional logic (e.g., "if the carry is 1, then add 1 to the next column"). Standard MLPs struggle to learn strict conditional logic. By adding a multiplicative gate, we give the model an internal "if/else" switch for every single neuron. The model can dynamically decide to turn certain features completely off or on depending on the current token context.

## What is the "Swi" in SwiGLU?
The term SwiGLU specifically refers to a GLU that uses the **Swish** activation function (also known as SiLU in PyTorch). 

There are many variants of GLU depending on the activation function used for the gate:
- **ReGLU**: Uses ReLU (Rectified Linear Unit)
- **GEGLU**: Uses GELU (Gaussian Error Linear Unit)
- **SwiGLU**: Uses Swish (Sigmoid Linear Unit)

In 2020, Noam Shazeer empirically tested all of these variants and found that **SwiGLU consistently outperformed all others**.

## Parameter Math
Because SwiGLU uses three linear layers instead of two, a naive implementation would result in $50\%$ more parameters than a standard MLP! To ensure a fair comparison and keep the model size the same, the industry standard is to scale the hidden dimension by $\frac{8}{3}$ instead of $4$.

**Standard MLP:**
- Hidden dim = $4 \times d$
- Parameters = $(d \times 4d) + (4d \times d) = 8d^2$

**SwiGLU:**
- Hidden dim = $\frac{8}{3} d$
- Parameters = $3 \times (d \times \frac{8}{3} d) = 8d^2$

*(As you can see, the parameter counts are mathematically identical!)*

## Industry Adoption
SwiGLU has become the gold standard for state-of-the-art open models. It is the primary FeedForward block used in:
- LLaMA (1, 2, and 3)
- Mistral / Mixtral
- Google PaLM / Gemma
