import sys
import os
import torch
import json
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from data.arc_dataset import ARCDataset, ARCTokenizer
from models.bdh import BDH, BDHConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(ckpt_path):
    # Same config as training
    config = BDHConfig(
        vocab_size=13, n_layer=8, n_embd=256, n_head=4, mlp_internal_dim_multiplier=64
    )
    model = BDH(config).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model

def generate_completion(model, start_tokens, max_new_tokens=1000):
    """
    Generates tokens until TASK_SEP or max_new_tokens.
    """
    tokenizer = ARCTokenizer()
    curr_tokens = torch.tensor([start_tokens], dtype=torch.long).to(DEVICE)
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # BDH forward
            logits, _ = model(curr_tokens)
            next_token_logits = logits[0, -1, :]
            
            # Greedy decoding
            next_token = torch.argmax(next_token_logits).item()
            
            # Stop condition
            if next_token == tokenizer.TASK_SEP:
                break
                
            generated.append(next_token)
            
            # Append to input for next step
            curr_tokens = torch.cat([
                curr_tokens, 
                torch.tensor([[next_token]], device=DEVICE)
            ], dim=1)
            
    return generated

def run_eval():
    ckpt_path = "results/bdh_arc_3000.pt" # Update this to your actual checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    model = load_model(ckpt_path)
    tokenizer = ARCTokenizer()
    
    # Load evaluation set
    with open('data/arc_evaluation.json') as f:
        tasks = json.load(f)
        
    print(f"Evaluating on {len(tasks)} tasks...")
    
    # Evaluate first 5 for now
    for i, (task_id, task) in enumerate(list(tasks.items())[:5]):
        print(f"Task {task_id}")
        
        # Prepare context (Train demos + Test Input)
        # We manually construct this because we need to STOP before the test output
        context_tokens = []
        for demo in task['train']:
            context_tokens.extend(tokenizer.encode_grid(demo['input']))
            context_tokens.append(tokenizer.IO_SEP)
            context_tokens.extend(tokenizer.encode_grid(demo['output']))
            context_tokens.append(tokenizer.TASK_SEP)
            
        context_tokens.extend(tokenizer.encode_grid(task['test'][0]['input']))
        context_tokens.append(tokenizer.IO_SEP)
        
        # Generate
        pred_tokens = generate_completion(model, context_tokens)
        
        # Compare (Naive exact match)
        gt_grid = task['test'][0]['output']
        gt_tokens = tokenizer.encode_grid(gt_grid)
        
        print(f"  Pred len: {len(pred_tokens)} | GT len: {len(gt_tokens)}")
        if pred_tokens == gt_tokens:
            print("  ✅ CORRECT")
        else:
            print("  ❌ INCORRECT")

if __name__ == "__main__":
    run_eval()
