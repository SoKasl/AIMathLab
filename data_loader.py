import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datasets import load_dataset
import random

# We use tiktoken, the official OpenAI tokenizer library, to convert text to integers
# You may need to run: pip install tiktoken datasets
tokenizer = tiktoken.get_encoding("gpt2")

class WikiTextDataset(Dataset):
    """
    Loads a small web/wiki dataset for pretraining.
    We stream it so it doesn't crash your Mac's RAM!
    """
    def __init__(self, split='train', block_size=256, max_samples=20000):
        super().__init__()
        self.block_size = block_size
        
        print(f"Loading WikiText {split} dataset...")
        # We load a small, high-quality Wikipedia dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split, streaming=True)
        
        self.data = []
        # We grab a chunk of text, tokenize it, and store it
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            text = item['text']
            if len(text) > 100: # skip very short or empty lines
                tokens = tokenizer.encode(text)
                self.data.extend(tokens)
                
        # Convert our massive list of integers into a PyTorch tensor
        self.data = torch.tensor(self.data, dtype=torch.long)
        print(f"Loaded {len(self.data)} tokens for pretraining.")

    def __len__(self):
        # The number of possible sequences we can grab
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of text (idx)
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # 'x' is the input context
        x = chunk[:-1]
        # 'y' is the target (x shifted by 1 into the future)
        y = chunk[1:]
        
        return x, y

class SyntheticMathDataset(Dataset):
    def __init__(self, filename, block_size=256):
        self.block_size = block_size
        self.data = []
        
        print(f"Loading synthetic math dataset from {filename}...")
        
        # Read the persisted text file
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Tokenize the entire text block
        self.data = tokenizer.encode(text)
        self.data = torch.tensor(self.data, dtype=torch.long)
        print(f"Loaded {len(self.data)} tokens from {filename}.")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloaders(batch_size=4, block_size=256):
    """ Helper function to easily get our dataloaders in the training script """
    wiki_dataset = WikiTextDataset(split='train', block_size=block_size)
    
    # Load our persisted math datasets
    math_train_dataset = SyntheticMathDataset("math_train.txt", block_size=block_size)
    math_eval_dataset = SyntheticMathDataset("math_eval.txt", block_size=block_size)
    
    # DataLoader handles batching (grabbing 4 sequences at a time) and shuffling
    wiki_loader = DataLoader(wiki_dataset, batch_size=batch_size, shuffle=True)
    math_train_loader = DataLoader(math_train_dataset, batch_size=batch_size, shuffle=True)
    math_eval_loader = DataLoader(math_eval_dataset, batch_size=batch_size, shuffle=True)
    
    return wiki_loader, math_train_loader, math_eval_loader

if __name__ == "__main__":
    # Test the dataloaders if you run this script directly
    wiki_loader, math_train_loader, math_eval_loader = get_dataloaders(batch_size=2, block_size=10)
    
    x, y = next(iter(math_train_loader))
    print("\n--- Synthetic Math Train DataLoader Test ---")
    print(f"Input shape (x): {x.shape}")
    print(f"Target shape (y): {y.shape}")
    
    print("\nFirst batch input (x):")
    print(tokenizer.decode(x[0].tolist()))
    print("First batch target (y):")
    print(tokenizer.decode(y[0].tolist()))
