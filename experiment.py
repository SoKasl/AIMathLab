import torch
import torch.optim as optim
import tiktoken
import csv
import time
import argparse
from data_loader import get_dataloaders
from evaluate import evaluate_model

def run_experiment(args):
    # Conditionally import model to keep base clean
    if args.use_kv_cache:
        from model_kv_cache import GPT, config
    else:
        from model import GPT, config
        
    # Enable features based on argparse flags
    config.use_flash_attention = args.flash_attention
    config.use_flash_attention_educational = args.flash_attention_educational
    config.use_swiglu = args.swiglu
    config.use_rmsnorm = args.rmsnorm
    config.use_rope = args.rope
    config.use_moe = args.moe
    config.use_kv_cache = args.use_kv_cache
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    wiki_intervals = sorted(args.wiki_intervals)
    math_eval_points = sorted(args.math_eval_points)
    
    # Initialize CSV
    csv_file = args.csv_output
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Wiki_Steps", "Math_Steps", "Elapsed_Time_sec", "Overall_Acc", "Add_Acc", "Sub_Acc", "Mul_Acc", "Div_Acc"])
        
    print(f"Starting experiment! Logging to {csv_file}")
    print(f"Flash Attention Enabled: {config.use_flash_attention}")
    print(f"Educational Flash Attention Enabled: {config.use_flash_attention_educational}")
    print(f"SwiGLU Enabled: {config.use_swiglu}")
    print(f"RMSNorm Enabled: {config.use_rmsnorm}")
    print(f"RoPE Enabled: {config.use_rope}")
    print(f"MoE Enabled: {config.use_moe}")
    start_time = time.time()
    
    for wiki_step in wiki_intervals:
        print(f"\n{'='*50}\nStarting new model: {wiki_step} WikiText Steps\n{'='*50}")
        
        # 1. Initialize a fresh model
        model = GPT().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        
        # 2. Get fresh dataloaders
        # Keeping block size at 256 for an exact 1:1 architectural comparison
        b_size = 256
        wiki_loader, math_train_loader, _ = get_dataloaders(
            batch_size=4, 
            block_size=b_size, 
            use_old_data=args.use_old_math_data, 
            use_curriculum_data=args.use_curriculum_data,
            use_hierarchical_data=args.use_hierarchical_data
        )
        
        # 3. Pretrain on WikiText
        model.train()
        wiki_iter = iter(wiki_loader)
        for step in range(wiki_step):
            try:
                x, y = next(wiki_iter)
            except StopIteration:
                wiki_iter = iter(wiki_loader)
                x, y = next(wiki_iter)
                
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x, y)
            logits, loss = outputs[0], outputs[1]
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 1000 == 0 or (step + 1) == wiki_step:
                print(f"[Wiki Pretraining] Step {step + 1}/{wiki_step} | Loss: {loss.item():.4f}")
                
        # 4. Math Alignment (Cumulative)
        current_math_step = 0
        math_iter = iter(math_train_loader)
        
        best_acc = -1.0
        best_model_path = csv_file.replace(".csv", f"_best_model_wiki{wiki_step}.pt")
        
        for eval_point in math_eval_points:
            # Train up to the next eval point
            steps_to_train = eval_point - current_math_step
            if steps_to_train > 0:
                print(f"\nTraining on Math from step {current_math_step} to {eval_point}...")
                model.train()
                for _ in range(steps_to_train):
                    try:
                        x, y = next(math_iter)
                    except StopIteration:
                        math_iter = iter(math_train_loader)
                        x, y = next(math_iter)
                        
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = model(x, y)
                    logits, loss = outputs[0], outputs[1]
                    loss.backward()
                    optimizer.step()
                    
                    current_math_step += 1
                    if current_math_step % 500 == 0:
                        print(f"[Math Alignment] Step {current_math_step}/{max(math_eval_points)} | Loss: {loss.item():.4f}")
            
            # Evaluate at the current eval point
            print(f"\nEvaluating Math at step {current_math_step} for {wiki_step} Wiki steps...")
            
            sample_log_path = csv_file.replace(".csv", f"_samples_wiki{wiki_step}_math{current_math_step}.txt")
            accs = evaluate_model(model, tokenizer, device, use_old_data=args.use_old_math_data, use_hierarchical_data=args.use_hierarchical_data, sample_log_path=sample_log_path)
            
            elapsed = time.time() - start_time
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([wiki_step, current_math_step, f"{elapsed:.2f}", accs['overall'], accs['+'], accs['-'], accs['*'], accs['/']])
            
            # Save the best model
            if accs['overall'] > best_acc:
                best_acc = accs['overall']
                print(f"New best accuracy {best_acc}%! Saving model to {best_model_path}")
                torch.save(model.state_dict(), best_model_path)
            
    # Save the final model weights just in case
    model_save_path = csv_file.replace(".csv", "_final.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nExperiment Complete! Results saved to {csv_file}")
    print(f"Final model weights saved to {model_save_path}")
    print(f"Best model weights saved to {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Math Training Experiment")
    parser.add_argument("--flash_attention", action="store_true", help="Use PyTorch optimized flash attention")
    parser.add_argument("--flash_attention_educational", action="store_true", help="Use educational flash attention")
    parser.add_argument("--swiglu", action="store_true", help="Use SwiGLU activation instead of standard GELU FeedForward")
    parser.add_argument("--rmsnorm", action="store_true", help="Use RMSNorm instead of standard LayerNorm")
    parser.add_argument("--rope", action="store_true", help="Use Rotary Position Embeddings (RoPE) instead of absolute pos embeddings")
    parser.add_argument("--moe", action="store_true", help="Use Sparse Mixture of Experts (MoE)")
    parser.add_argument("--csv_output", type=str, default="experiment_results.csv", help="Output CSV filename")
    parser.add_argument("--use_old_math_data", action="store_true", help="Use the old non-CoT math datasets")
    parser.add_argument("--use_curriculum_data", action="store_true", help="Use the new curriculum math datasets")
    parser.add_argument("--use_hierarchical_data", action="store_true", help="Use the 4-stage hierarchical developmental curriculum math dataset")
    parser.add_argument("--use_kv_cache", action="store_true", help="Use KV cache during inference/evaluation")
    parser.add_argument("--wiki_intervals", nargs="+", type=int, default=[100, 5000, 10000, 20000], help="List of Wiki steps")
    parser.add_argument("--math_eval_points", nargs="+", type=int, default=[0, 1000, 2000, 3000, 4000, 5000], help="List of Math eval points")
    args = parser.parse_args()
    
    run_experiment(args)
