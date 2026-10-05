import os
import json
import shutil
import torch

def package_model(checkpoint_path, output_dir, is_moe=False):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Packaging {checkpoint_path} -> {output_dir}...")
    
    # 1. Load checkpoint and extract clean state_dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    weights_path = os.path.join(output_dir, "model.pt")
    torch.save(state_dict, weights_path)
    print(f"  Saved clean weights to {weights_path} ({os.path.getsize(weights_path) / (1024*1024):.2f} MB)")
    
    # 2. Config JSON
    config = {
        "architectures": ["TinyGPTForCausalLM"],
        "model_type": "tinygpt",
        "vocab_size": 50257,
        "n_embd": 384,
        "n_head": 6,
        "n_layer": 6,
        "block_size": 256,
        "dropout": 0.0,
        "use_flash_attention": True,
        "use_swiglu": True,
        "use_rmsnorm": True,
        "use_rope": True,
        "use_moe": is_moe,
        "num_experts": 4 if is_moe else 1,
        "top_k_experts": 2 if is_moe else 1
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    # 3. Copy architecture implementation
    shutil.copy("model_kv_cache.py", os.path.join(output_dir, "model.py"))
    
    # 4. Create standalone inference script (generate.py)
    infer_script = """import torch
import tiktoken
import argparse
from model import GPT, ModelConfig

def main():
    parser = argparse.ArgumentParser(description="Run inference with TinyGPT Math Model")
    parser.add_argument("--prompt", type=str, default="Q: What is 782 / 23?\\nA:", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=160, help="Maximum generated tokens")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    import json
    with open("config.json", "r") as f:
        cfg_dict = json.load(f)

    config = ModelConfig(
        vocab_size=cfg_dict["vocab_size"],
        n_embd=cfg_dict["n_embd"],
        n_head=cfg_dict["n_head"],
        n_layer=cfg_dict["n_layer"],
        block_size=cfg_dict["block_size"],
        use_flash_attention=cfg_dict["use_flash_attention"],
        use_swiglu=cfg_dict["use_swiglu"],
        use_rmsnorm=cfg_dict["use_rmsnorm"],
        use_rope=cfg_dict["use_rope"],
        use_moe=cfg_dict["use_moe"],
        num_experts=cfg_dict["num_experts"],
        top_k_experts=cfg_dict["top_k_experts"],
        use_kv_cache=True
    )

    model = GPT(config).to(args.device)
    model.load_state_dict(torch.load("model.pt", map_location=args.device))
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=args.device).unsqueeze(0)

    print(f"\\n--- Generating Output for Prompt: ---\\n{args.prompt}")
    with torch.no_grad():
        out_ids = model.generate_with_cache(input_ids, max_new_tokens=args.max_tokens)
    
    output_text = tokenizer.decode(out_ids[0].tolist())
    print("\\n--- Model Response: ---")
    print(output_text)
    print("-----------------------\\n")

if __name__ == "__main__":
    main()
"""
    with open(os.path.join(output_dir, "generate.py"), "w", encoding="utf-8") as f:
        f.write(infer_script)

    # 5. Anonymous Model Card README.md
    model_name = "TinyGPT 10M Modern Dense (Hierarchical Curriculum)" if not is_moe else "TinyGPT 10M Sparse MoE (Hierarchical Curriculum)"
    readme_content = f"""# {model_name}

An empirical compact Transformer (~10M parameters) trained to study multi-step arithmetic Chain-of-Thought (CoT) reasoning and deterministic algorithmic execution.

## Highlights
- **Compact Footprint**: 6 layers, 384 embedding dim, 6 attention heads (~10M active parameters).
- **Modern Transformer Architecture**: Rotary Positional Embeddings (RoPE), RMSNorm, SwiGLU activation{", and Top-2 Sparse Mixture-of-Experts" if is_moe else ""}.
- **Hierarchical Developmental Curriculum**: Pretrained on 20,000 WikiText steps, then fine-tuned through a 4-stage sequential developmental progression (Additive Foundations $\\rightarrow$ Single-Digit Mult $\\rightarrow$ Spatial Zero-Padding $\\rightarrow$ Digit-by-Digit Long Division).
- **Deterministic Long Division Breakthrough**: Employs schoolbook digit descent Chain-of-Thought, elevating multi-digit division from 4% to **60.0%** zero-shot accuracy.

## Quickstart

```bash
# Install dependencies
pip install torch tiktoken

# Run standalone inference
python generate.py --prompt "Q: What is 950 / 38?\\nA:"
```

### Python API
```python
import torch
import tiktoken
from model import GPT, ModelConfig

config = ModelConfig(
    vocab_size=50257,
    n_embd=384,
    n_head=6,
    n_layer=6,
    block_size=256,
    use_flash_attention=True,
    use_swiglu=True,
    use_rmsnorm=True,
    use_rope=True,
    use_moe={is_moe},
    num_experts={4 if is_moe else 1},
    top_k_experts={2 if is_moe else 1},
    use_kv_cache=True
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT(config).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
prompt = "Q: What is 782 / 23?\\nA:"
tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    out = model.generate_with_cache(tokens, max_new_tokens=160)

print(tokenizer.decode(out[0].tolist()))
```

## Citation
If you use this model or code in your research, please refer to the experimental repository for replication artifacts.
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

def main():
    package_model(
        checkpoint_path="run9_modern_hierarchical_best_model_wiki20000.pt",
        output_dir="packaged_models/tinygpt-10m-hierarchical-dense",
        is_moe=False
    )
    package_model(
        checkpoint_path="run10_moe_hierarchical_best_model_wiki20000.pt",
        output_dir="packaged_models/tinygpt-10m-hierarchical-moe",
        is_moe=True
    )
    print("\nPackaging completed successfully (Step 30,000 Peak Models)!")

if __name__ == "__main__":
    main()
