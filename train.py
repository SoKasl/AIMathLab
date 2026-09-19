import torch
from model import GPT
from data_loader import get_dataloaders, tokenizer

# 1. Setup the Device (The GPU Magic)
# This is where we tell PyTorch to use your M4 chip's GPU!
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# 2. Initialize the Model
print("Initializing the Mini-GPT model...")
model = GPT()
model = model.to(device)

# We use AdamW, the standard optimizer for training LLMs
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Load our data
batch_size = 8
wiki_loader, gsm8k_loader = get_dataloaders(batch_size=batch_size, block_size=256)

def train_loop(dataloader, steps, phase_name):
    """ A simple training loop """
    model.train() # Tell the model it is in training mode
    
    print(f"\n--- Starting {phase_name} Phase ---")
    data_iter = iter(dataloader)
    
    for step in range(steps):
        # Grab a batch of (x, y) data
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
            
        # Send the data to the GPU
        x, y = x.to(device), y.to(device)
        
        # 1. Forward Pass (Ask the model to guess)
        logits, loss = model(x, targets=y)
        
        # 2. Backward Pass (Calculate how to fix the mistakes)
        optimizer.zero_grad() # Clear old memory
        loss.backward()       # Do the heavy math
        optimizer.step()      # Update the weights!
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

def generate_text(prompt, max_tokens=20):
    """ Let the model write some text! """
    model.eval() # Put model in 'testing' mode
    
    # Convert our text prompt into integers and send to GPU
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    # Generate new tokens (this returns the prompt AND the new tokens)
    generated_idx = model.generate_next_token(idx, max_new_tokens=max_tokens)
    
    # Convert integers back to text
    output_text = tokenizer.decode(generated_idx[0].tolist())
    print(output_text)
    print("-" * 30) # Add a separator line for readability

if __name__ == "__main__":
    print("\n--- Testing Untrained Model ---")
    generate_text("The secret to life is", max_tokens=10)
    
    # Pretrain on Wikipedia for a few steps
    train_loop(wiki_loader, steps=50, phase_name="Pretraining (WikiText)")
    
    print("\n--- Testing Before Math Training ---")
    equation = "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA:"
    generate_text(equation, max_tokens=10)

    # Align on Math problems for a few steps
    train_loop(gsm8k_loader, steps=50, phase_name="Alignment (GSM8K Math)")
    
    print("\n--- Testing After Math Training ---")
    generate_text(equation, max_tokens=10)
