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
            
        # Parse problem by problem and pack them into dense 256-token blocks
        blocks = text.strip().split('\n\n')
        pad_token = 50256 # <|endoftext|> in GPT2
        
        parsed = []
        for b in blocks:
            b_str = b.strip()
            if not b_str: continue
            
            # Split into prompt (question) and completion (thought + answer)
            if '\nA:' in b_str:
                parts = b_str.split('\nA:')
                prompt_str = parts[0].strip() + '\nA:'
                completion_str = parts[1]
            else:
                prompt_str = b_str
                completion_str = ""
                
            p_tok = tokenizer.encode(prompt_str)
            # Completion includes the EOS token so the model learns when to terminate
            c_tok = tokenizer.encode(completion_str) + [pad_token]
            
            # Only include problems that can fit in a single block
            if len(p_tok) + len(c_tok) <= block_size + 1:
                parsed.append((p_tok, c_tok))
                
        # Pack complete problems into blocks of size block_size
        self.samples = []
        curr_tokens = []
        curr_masks = []
        
        for p_tok, c_tok in parsed:
            prob_len = len(p_tok) + len(c_tok)
            # If adding this problem exceeds block_size + 1, finalize the current block
            if len(curr_tokens) + prob_len > block_size + 1:
                pad_len = block_size - len(curr_tokens[:-1])
                x = curr_tokens[:-1] + [pad_token] * pad_len
                y = curr_masks + [-100] * pad_len
                self.samples.append((
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long)
                ))
                curr_tokens = []
                curr_masks = []
                
            offset = len(curr_tokens)
            curr_tokens.extend(p_tok + c_tok)
            
            # Target masks for this problem:
            # Mask prompt tokens in y with -100 so loss is NOT computed on the prompt
            if offset == 0:
                for _ in range(len(p_tok) - 1):
                    curr_masks.append(-100)
            else:
                # Transition token from previous problem's EOS to this problem's prompt -> mask with -100
                curr_masks.append(-100)
                for _ in range(len(p_tok) - 1):
                    curr_masks.append(-100)
            # Active loss on all completion tokens and EOS
            curr_masks.extend(c_tok)
            
        # Finalize any remaining tokens
        if curr_tokens:
            pad_len = block_size - len(curr_tokens[:-1])
            x = curr_tokens[:-1] + [pad_token] * pad_len
            y = curr_masks + [-100] * pad_len
            self.samples.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            ))
            
        print(f"Loaded {len(self.samples)} sequence-packed math blocks from {filename}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_dataloaders(batch_size=4, block_size=256, use_old_data=False, use_curriculum_data=False, use_hierarchical_data=False):
    """ Helper function to easily get our dataloaders in the training script """
    wiki_dataset = WikiTextDataset(split='train', block_size=block_size)
    
    shuffle_math = True
    if use_hierarchical_data:
        train_file = "math_hierarchical_curriculum_train.txt"
        eval_file = "math_eval_hierarchical.txt"
        # In hierarchical curriculum, preserve sequential order so the model learns
        # Stage 1 (Add/Sub) -> Stage 2 (1-digit Mul) -> Stage 3 (Multi-digit FOIL) -> Stage 4 (Division)
        shuffle_math = False
    elif use_old_data:
        train_file = "math_train_old.txt"
        eval_file = "math_eval_old.txt"
    elif use_curriculum_data:
        train_file = "math_train_curriculum.txt"
        eval_file = "math_eval_curriculum.txt"
    else:
        train_file = "math_train.txt"
        eval_file = "math_eval.txt"
    
    # Check data/ and data/train/ subdirectories if not in root
    if not os.path.exists(train_file):
        if os.path.exists(os.path.join("data", train_file)):
            train_file = os.path.join("data", train_file)
        elif os.path.exists(os.path.join("data", "train", train_file)):
            train_file = os.path.join("data", "train", train_file)
            
    if not os.path.exists(eval_file) and os.path.exists(os.path.join("data", eval_file)):
        eval_file = os.path.join("data", eval_file)

    # Load our persisted math datasets
    math_train_dataset = SyntheticMathDataset(train_file, block_size=block_size)
    math_eval_dataset = SyntheticMathDataset(eval_file, block_size=block_size)
    
    # DataLoader handles batching (grabbing 4 sequences at a time) and shuffling
    wiki_loader = DataLoader(wiki_dataset, batch_size=batch_size, shuffle=True)
    math_train_loader = DataLoader(math_train_dataset, batch_size=batch_size, shuffle=shuffle_math)
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
