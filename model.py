# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.nn import functional as F

from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.2
    
    use_flash_attention: bool = False
    use_flash_attention_educational: bool = False
    use_swiglu: bool = False
    use_rmsnorm: bool = False
    use_rope: bool = False
    use_moe: bool = False
    use_kv_cache: bool = False

# Global instance for backwards compatibility
config = ModelConfig()
vocab_size = config.vocab_size
n_embd = config.n_embd
n_head = config.n_head
n_layer = config.n_layer
block_size = config.block_size
dropout = config.dropout

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the complex frequencies for RoPE.
    Think of this as creating a 'clock face' of angles for every possible position 
    in the sequence (up to 'end').
    - dim: the dimension of the head (head_size)
    - end: max sequence length (config.block_size)
    Returns a tensor of complex numbers (polar coordinates) representing the rotations.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape the 1D precomputed frequencies so they can be broadcast 
    and multiplied across the higher-dimensional Query and Key tensors (Batch, Heads, SeqLen, HeadDim).
    """
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Takes the Query (xq) and Key (xk) tensors and rotates them by the precomputed frequencies.
    1. Reshapes the real numbers into pairs (2D coordinates) and casts them to Complex numbers.
    2. Multiplies them by the Complex frequencies (which mathematically rotates them).
    3. Casts them back to Real numbers and flattens them back to their original shape.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # Register a buffer for the causal mask (so the model can't look ahead)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        if config.use_rope:
            self.register_buffer('freqs_cis', precompute_freqs_cis(head_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        if self.config.use_rope:
            q = q.unsqueeze(2)
            k = k.unsqueeze(2)
            q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])
            q = q.squeeze(2)
            k = k.squeeze(2)
            
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ 
    Multiple heads of self-attention in parallel.
    
    EDUCATIONAL NOTE: Flash Attention
    ---------------------------------
    Classic attention computes a massive (Sequence_Length x Sequence_Length) matrix of probabilities.
    This takes O(N^2) memory and is extremely slow.
    Flash Attention is a "hardware-aware" optimization. It computes this exact same math directly 
    inside the GPU's ultra-fast SRAM in chunks, completely bypassing the slow main VRAM.
    Result: 2x-3x faster training and significantly less memory usage, with mathematically identical outputs!
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if not config.use_flash_attention and not config.use_flash_attention_educational:
            # --- CLASSIC SLOW ATTENTION SETUP ---
            head_size = config.n_embd // config.n_head
            self.heads = nn.ModuleList([Head(head_size, config) for _ in range(config.n_head)])
            self.proj = nn.Linear(head_size * config.n_head, config.n_embd)
            self.dropout = nn.Dropout(config.dropout)
        else:
            # --- FLASH / EDUCATIONAL ATTENTION SETUP ---
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.dropout_val = config.dropout
            
            # Instead of separate heads, we create one massive linear layer that computes
            # Query, Key, and Value for ALL heads simultaneously! 
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            self.dropout = nn.Dropout(config.dropout)
            if config.use_rope:
                self.register_buffer('freqs_cis', precompute_freqs_cis(config.n_embd // config.n_head, config.block_size))

    def forward(self, x):
        """ Routes to the correct attention implementation based on ModelConfig """
        if not self.config.use_flash_attention and not self.config.use_flash_attention_educational:
            return self._forward_classic(x)
        elif self.config.use_flash_attention:
            return self._forward_flash(x)
        else:
            return self._forward_flash_educational(x)

    def _forward_classic(self, x):
        """ CLASSIC: Loop through every head individually (slow) """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    def _forward_flash(self, x):
        """ FLASH: Compute everything at once in massive batches (fast) """
        B, T, C = x.size() # Batch size, Sequence Length, Embedding Dimension
        
        # 1. Calculate Q, K, V for all heads in one shot
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 2. Reshape into (Batch, Sequence Length, Heads, Head Size)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)
        
        if self.config.use_rope:
            q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])
            
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 3. Flash Attention Magic!
        # is_causal=True automatically handles the lower-triangular masking so the model can't look ahead.
        # We don't even need our 'tril' mask buffer anymore!
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                           dropout_p=self.dropout_val if self.training else 0.0, 
                                           is_causal=True)
        
        # 4. Re-assemble the heads back into a flat sequence
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 5. Final projection
        out = self.dropout(self.c_proj(y))
        return out

    def _forward_flash_educational(self, x):
        """ 
        EDUCATIONAL FLASH ATTENTION FROM SCRATCH
        This implements the block-tiling algorithm in pure Python so learners can read the math.
        WARNING: This will run much slower than Classic or Flash because Python is slow.
        """
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)
        
        if self.config.use_rope:
            q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])
            
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Tile sizes (blocks to load into "SRAM")
        Br = 32
        Bc = 32
        
        O = torch.zeros_like(q)
        l = torch.zeros((B, self.n_head, T, 1), device=q.device)
        m = torch.full((B, self.n_head, T, 1), float('-inf'), device=q.device)
        d = q.shape[-1]
        
        # Outer loop: Query blocks
        for i in range(0, T, Br):
            q_i = q[:, :, i:i+Br, :]
            
            # Inner loop: Key/Value blocks
            for j in range(0, T, Bc):
                # Skip blocks that are entirely in the future (causal mask logic)
                if j > i + Br - 1:
                    continue

                k_j = k[:, :, j:j+Bc, :]
                v_j = v[:, :, j:j+Bc, :]

                # 1. Compute block-wise affinities
                S_ij = q_i @ k_j.transpose(-2, -1) / (d ** 0.5)

                # 2. Apply precise block causal masking
                row_idx = torch.arange(i, i + q_i.shape[2], device=q.device).unsqueeze(1)
                col_idx = torch.arange(j, j + k_j.shape[2], device=q.device).unsqueeze(0)
                mask = row_idx >= col_idx
                S_ij = S_ij.masked_fill(~mask, float('-inf'))

                # 3. Online Softmax Trick (keep running maximums and sums)
                m_i_prev = m[:, :, i:i+Br, :]
                l_i_prev = l[:, :, i:i+Br, :]

                m_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
                m_i_new = torch.maximum(m_i_prev, m_ij)

                P_ij = torch.exp(S_ij - m_i_new)
                l_i_new = torch.exp(m_i_prev - m_i_new) * l_i_prev + torch.sum(P_ij, dim=-1, keepdim=True)

                # 4. Multiply by V and accumulate output
                O_i_prev = O[:, :, i:i+Br, :]
                O[:, :, i:i+Br, :] = torch.exp(m_i_prev - m_i_new) * O_i_prev + P_ij @ v_j

                m[:, :, i:i+Br, :] = m_i_new
                l[:, :, i:i+Br, :] = l_i_new

        # 5. Divide output by running sum to normalize
        y = O / l
        
        # Re-assemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.c_proj(y))
        return out

class SwiGLU(nn.Module):
    """ 
    Swish-Gated Linear Unit (SwiGLU)
    Replaces the standard MLP block. Uses a multiplicative gate with SiLU activation.
    """
    def __init__(self, config):
        super().__init__()
        # Industry standard: scale by 8/3 to keep parameters roughly identical to a 4x standard MLP
        hidden_dim = int(8 * config.n_embd / 3)
        # Often rounded to a multiple of 16 for hardware efficiency, but we'll stick to exact 8/3 for simplicity
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False) # Gate projection
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False) # Up projection
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False) # Down projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU(x, W1, W2, W3) = W2 * ( Swish(W1 * x) ⊗ (W3 * x) )
        # F.silu is PyTorch's implementation of the Swish activation function
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.dropout(self.w2(gate * up))

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization
    Replaces LayerNorm. Drops the mean-centering step for faster computation.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x.pow(2).mean() calculates the variance (mean of squared values)
        # We don't subtract the mean like LayerNorm does
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x_norm

class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation (feed-forward) """
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        
        # Route to SwiGLU if flag is set, else fallback to classic FeedForward
        if config.use_swiglu:
            self.ffwd = SwiGLU(config)
        else:
            self.ffwd = FeedForward(config)
            
        # Route to RMSNorm if flag is set, else fallback to classic LayerNorm
        NormLayer = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln1 = NormLayer(config.n_embd)
        self.ln2 = NormLayer(config.n_embd)

    def forward(self, x):
        # Residual connections around the attention and feed-forward
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    """ The complete GPT Language Model """
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is None:
            model_config = config # Use global config fallback
        self.config = model_config
        
        # Token embedding and positional embedding
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding_table = nn.Embedding(self.config.block_size, self.config.n_embd)
        
        # Sequence of transformer blocks
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        
        NormLayer = RMSNorm if self.config.use_rmsnorm else nn.LayerNorm
        self.ln_f = NormLayer(self.config.n_embd) # final normalization
        
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size) # outputs probability distribution over vocab
        
        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        if not self.config.use_rope:
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
            x = tok_emb + pos_emb # Combine token and position information
        else:
            x = tok_emb
            
        x = self.blocks(x) # Pass through transformer blocks
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #2D array for cross entropy loss function
            targets = targets.view(B*T) #1D array for targets
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate_next_token(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
