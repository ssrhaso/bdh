import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path so we can import from models
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from data.arc_dataset import ARCDataset
from models.bdh import BDH, BDHConfig 

# --- Config ---
# You can move this to a yaml file later, but let's keep it simple for now
CONFIG = {
    'batch_size': 4,           # Small batch size for large VRAM usage
    'max_iters': 10000,
    'lr': 3e-4,
    'eval_interval': 500,
    'save_interval': 1000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def train():
    device = CONFIG['device']
    print(f"Starting training on {device}...")
    
    # 1. Load Data
    # Make sure you have the ARC json files in data/
    train_dataset = ARCDataset('data/arc_training.json', max_length=4096)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 2. Init Model
    # Using specific BDH params for ARC (Reasoning requires depth)
    bdh_cfg = BDHConfig(
        vocab_size=13,
        n_layer=8,
        n_embd=256,
        n_head=4,
        mlp_internal_dim_multiplier=64, # 4096 neurons
        dropout=0.05
    )
    
    model = BDH(bdh_cfg).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    # 4. Loop
    model.train()
    iter_num = 0
    
    while iter_num < CONFIG['max_iters']:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass (Official BDH calculates loss if targets are passed)
            logits, loss = model(x, targets=y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            iter_num += 1
            
            if iter_num % 10 == 0:
                print(f"Iter {iter_num} | Loss: {loss.item():.4f}")
                
            if iter_num % CONFIG['save_interval'] == 0:
                ckpt_path = f"results/bdh_arc_{iter_num}.pt"
                os.makedirs("results", exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
                
            if iter_num >= CONFIG['max_iters']:
                break

if __name__ == "__main__":
    train()
