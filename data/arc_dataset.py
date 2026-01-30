"""
ARC Dataset for BDH-GPU.
Converts 2D ARC grids into 1D token sequences for next-token prediction.
"""

import json
import torch
from torch.utils.data import Dataset

class ARCTokenizer:
    """
    Simple tokenizer for ARC grids.
    Tokens 0-9: Colors
    Token 10: Row Separator (ROW_SEP)
    Token 11: Input/Output Separator (IO_SEP)
    Token 12: Task Separator (TASK_SEP)
    """
    def __init__(self):
        self.vocab_size = 13
        self.ROW_SEP = 10
        self.IO_SEP = 11
        self.TASK_SEP = 12

    def encode_grid(self, grid):
        """Flattens a 2D grid into a 1D token list with row separators."""
        tokens = []
        for row in grid:
            tokens.extend(row)
            tokens.append(self.ROW_SEP)
        return tokens[:-1] if tokens else [] # Remove trailing row sep

    def encode_task(self, task):
        """
        Encodes a full ARC task into a single sequence:
        [Demo1_In] [IO] [Demo1_Out] [TASK] ... [Test_In] [IO] [Test_Out]
        """
        tokens = []
        
        # Training examples
        for demo in task['train']:
            tokens.extend(self.encode_grid(demo['input']))
            tokens.append(self.IO_SEP)
            tokens.extend(self.encode_grid(demo['output']))
            tokens.append(self.TASK_SEP)
            
        # Test example (Input only, model predicts output)
        if 'test' in task and len(task['test']) > 0:
            tokens.extend(self.encode_grid(task['test'][0]['input']))
            tokens.append(self.IO_SEP)
            if 'output' in task['test'][0]:
                tokens.extend(self.encode_grid(task['test'][0]['output']))
                
        return tokens

class ARCDataset(Dataset):
    def __init__(self, json_path, max_length=4096):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.tasks = list(data.values())
        self.tokenizer = ARCTokenizer()
        self.max_length = max_length
        print(f"Loaded {len(self.tasks)} tasks from {json_path}")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        tokens = self.tokenizer.encode_task(task)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            
        # Create inputs (x) and targets (y)
        # y is x shifted by 1 to the left
        # Example: x=[A, B, C], y=[B, C, D]
        
        # Pad with 0s to fixed length
        padding_len = self.max_length - len(tokens)
        padded_tokens = tokens + [0] * padding_len
        
        # Convert to tensor
        data = torch.tensor(padded_tokens, dtype=torch.long)
        
        x = data[:-1]
        y = data[1:]
        
        # Mask loss for padding (optional, but good practice)
        # We'll handle this simply by letting the model learn 0->0 mapping for now
        
        return x, y
