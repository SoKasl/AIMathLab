import torch
import torch.optim as optim
import tiktoken
import csv
import time
from model import GPT
from data_loader import get_dataloaders
from evaluate import evaluate_model

def run_experiment():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    wiki_intervals = [100, 5000, 10000, 15000, 20000]
    math_eval_points = [0, 1000, 2000, 3000, 4000, 5000]
    
    # Initialize CSV
    csv_file = "experiment_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Wiki_Steps", "Math_Steps", "Elapsed_Time_sec", "Overall_Acc", "Add_Acc", "Sub_Acc", "Mul_Acc", "Div_Acc"])
        
    print(f"Starting experiment! Logging to {csv_file}")
    start_time = time.time()
    
    for wiki_step in wiki_intervals:
        print(f"\n{'='*50}\nStarting new model: {wiki_step} WikiText Steps\n{'='*50}")
        
        # 1. Initialize a fresh model
        model = GPT().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        
        # 2. Get fresh dataloaders
        wiki_loader, math_train_loader, _ = get_dataloaders(batch_size=4, block_size=256)
        
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
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 1000 == 0 or (step + 1) == wiki_step:
                print(f"[Wiki Pretraining] Step {step + 1}/{wiki_step} | Loss: {loss.item():.4f}")
                
        # 4. Math Alignment (Cumulative)
        current_math_step = 0
        math_iter = iter(math_train_loader)
        
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
                    logits, loss = model(x, y)
                    loss.backward()
                    optimizer.step()
                    
                    current_math_step += 1
                    if current_math_step % 500 == 0:
                        print(f"[Math Alignment] Step {current_math_step}/{max(math_eval_points)} | Loss: {loss.item():.4f}")
            
            # Evaluate at the current eval point
            print(f"\nEvaluating Math at step {current_math_step} for {wiki_step} Wiki steps...")
            accs = evaluate_model(model, tokenizer, device)
            
            elapsed = time.time() - start_time
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([wiki_step, current_math_step, f"{elapsed:.2f}", accs['overall'], accs['+'], accs['-'], accs['*'], accs['/']])
            
    print(f"\nExperiment Complete! Results saved to {csv_file}")

if __name__ == "__main__":
    run_experiment()
