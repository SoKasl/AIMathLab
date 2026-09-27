import torch
import time
from model_kv_cache import GPT, ModelConfig

def test_kv_cache():
    print("Setting up model for KV cache testing...")
    # Initialize a small test model with Flash Attention and RoPE
    config = ModelConfig(
        vocab_size=1000, 
        n_embd=256, 
        n_head=4, 
        n_layer=4, 
        block_size=256,
        use_flash_attention=True,
        use_rope=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = GPT(config).to(device)
    model.eval()

    # Create a dummy input prompt
    prompt = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    max_new_tokens = 100

    print(f"Testing generation on device: {device}")
    
    # Run WITHOUT cache
    torch.manual_seed(42)
    start_time = time.time()
    out_no_cache = model.generate_next_token(prompt, max_new_tokens)
    time_no_cache = time.time() - start_time
    print(f"Without Cache: {time_no_cache:.4f} seconds")

    # Run WITH cache
    torch.manual_seed(42)
    start_time = time.time()
    out_with_cache = model.generate_with_cache(prompt, max_new_tokens)
    time_with_cache = time.time() - start_time
    print(f"With Cache: {time_with_cache:.4f} seconds")

    # Verify identical output
    is_identical = torch.equal(out_no_cache, out_with_cache)
    print(f"Outputs Identical: {is_identical}")
    
    if is_identical:
        speedup = time_no_cache / time_with_cache
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("ERROR: Outputs mismatch!")
        print("No cache:", out_no_cache[0, -10:])
        print("With cache:", out_with_cache[0, -10:])

if __name__ == "__main__":
    test_kv_cache()
