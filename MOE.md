# Sparse Mixture of Experts (MoE)

This repository implements a **Sparse Mixture of Experts (MoE)** architecture, designed to scale model capacity (parameters) without scaling compute costs.

## The Problem with Dense Models
In a standard Transformer block, every token in a sequence passes through the same dense FeedForward network (or SwiGLU). To make the model smarter, you must make the FeedForward layer larger. However, doing so increases the computational cost for *every single token*. 

## The MoE Solution
MoE replaces the single, monolithic FeedForward layer with $N$ smaller FeedForward networks, called **"Experts"**. 
Instead of sending every token through a massive layer, a **Router** network decides which $K$ experts are best suited to process each specific token.

For example, if $N=4$ and $K=2$:
1. The Router analyzes a token and outputs probabilities for all 4 experts.
2. It selects the top 2 experts for that token.
3. The token is passed *only* to those 2 experts.
4. The outputs from those 2 experts are combined using a weighted sum (based on the router's probabilities).

### Why "Sparse"?
It is "sparse" because the model only activates a small subset (e.g., 2 out of 4) of its parameters for any given token. If you have 8 experts but only route to 2, your model has 4x the total parameter capacity of a dense model, but runs at roughly the same speed!

## Implementation Details in `model.py`

### 1. The `SparseMoE` Class
The `SparseMoE` module acts as a drop-in replacement for the standard FeedForward/SwiGLU block.

```python
class SparseMoE(nn.Module):
    def __init__(self, config):
        ...
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        
        # 1. Initialize N experts
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(self.num_experts)])
        
        # 2. The Router (Linear projection)
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)
```

### 2. The Routing Logic (`forward` pass)
1. **Logits**: The router calculates a score for each expert `(B*T, num_experts)`.
2. **Top-K**: We use `torch.topk` to find the indices of the `top_k_experts` for each token.
3. **Softmax Weights**: The router probabilities for the chosen experts are normalized so they sum to 1.
4. **Dispatch**: We iterate through each expert. For each expert, we gather the tokens assigned to it, process them, and add them back to the final output tensor multiplied by their respective routing weights.

```python
# Gather tokens assigned to this expert
expert_mask = (selected_experts == i)
token_indices = torch.nonzero(expert_mask.any(dim=-1), as_tuple=True)[0]

# Process tokens through the expert
expert_outputs = expert(x_flat[token_indices])

# Add to final output (weighted)
final_output[token_indices] += expert_outputs * weights
```

### 3. How does the Router learn? (Training vs Inference)
The routing mechanism works **identically** during both training and inference. 

The router is a fully differentiable neural network layer (`nn.Linear`). Because the final output of the MoE layer multiplies the expert's computation by the router's probability (`expert_outputs * weights`), the backpropagation algorithm is able to send error gradients backward through those weights directly into the router.

Over thousands of training steps, the router uses gradient descent to learn which experts minimize the loss for specific types of tokens. It might naturally evolve to send all math operators to Expert 1, and all punctuation to Expert 2!

*(Note: Enterprise-scale MoE models like Mixtral often add a "Load Balancing Loss" penalty during training to prevent the router from getting lazy and sending every single token to the exact same expert, which would cause the other experts to "starve". We omitted this from our implementation for educational simplicity).*

## Running with MoE
You can enable this architecture during experiments by passing the `--moe` flag:

```bash
python experiment.py --moe --swiglu --rmsnorm --rope
```

By combining MoE with SwiGLU, RMSNorm, and RoPE, we create a highly capable, modern Transformer that can learn complex mathematical reasoning (like FOIL) much more efficiently than a classic dense GPT-2!
