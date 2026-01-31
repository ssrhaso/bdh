"""
Multi-Hop Variable Tracking Dataset Generator
(Generates synthetic reasoning tasks for probing length generalization)

Task: Variable assignment chains with query resolution
Example: a = b, b = c, c = 5. Query: a? Answer: 5

TASK INSPIRATION & CITATIONS:
This task is inspired by several established reasoning benchmarks:

1. Variable Binding / Pointer Chasing: Tests transitive reasoning through
   variable indirection. Similar to tasks in:
   - Zhang et al. (2021) "Pointer Value Retrieval: A New Benchmark for 
     Understanding the Limits of Neural Network Generalization"
     https://arxiv.org/abs/2107.12580

2. Compositional Generalization: Tests whether models can generalize
   to longer reasoning chains than seen during training:
   - Lake & Baroni (2018) "Generalization without Systematicity: On the 
     Compositional Skills of Sequence-to-Sequence Recurrent Networks"
     https://arxiv.org/abs/1711.00350
   - Hupkes et al. (2020) "Compositionality Decomposed: How do Neural 
     Networks Generalise?" https://arxiv.org/abs/1908.08351

3. Length Generalization in Transformers: Probes known failure modes:
   - Anil et al. (2022) "Exploring Length Generalization in Large Language 
     Models" https://arxiv.org/abs/2207.04901
   - Press et al. (2022) "Train Short, Test Long: Attention with Linear 
     Biases Enables Input Length Extrapolation" (ALiBi)
     https://arxiv.org/abs/2108.12409

The specific formulation (variable chains with terminal value) is a simplified 
version of pointer value retrieval that isolates the length generalization
phenomenon without confounding factors like natural language understanding.
"""

import torch
import random
from typing import Tuple

# VOCAB DEFINITIONS (CONSTANTS)
PAD_TOKEN = 32    # PADDING (for fixed-length sequences)
ASSIGN_TOKEN = 30 # '=' (assignment operator)
QUERY_TOKEN = 31  # '?' (query operator)
VAR_OFFSET = 10   # VARIABLES start at token ID 10 (e.g., v0=10, v1=11, ...)
VAR_COUNT = 20    # VARIABLES: tokens 10-29 (20 unique variables)
VALUE_RANGE = 10  # VALUES: tokens 0-9 (final answers)


def generate_reasoning_sample(
    chain_length: int = 3,
    vocab_offset: int = VAR_OFFSET
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GENERATE A SINGLE REASONING SAMPLE WITH A VARIABLE ASSIGNMENT CHAIN.
    
    (e.g., for chain_length=3: v0 = v1, v1 = v2, v2 = 5. Query: v0?)
    """
    
    # UNIQUE VARIABLES SELECTED
    vars_pool = list(range(vocab_offset, vocab_offset + VAR_COUNT))
    chain_vars = random.sample(vars_pool, min(chain_length, VAR_COUNT))
    
    # RANDOM FINAL VALUE (0-9)
    final_value = random.randint(0, VALUE_RANGE - 1)
    
    # BUILD TOKEN SEQUENCE
    tokens = []
    
    # ASSIGNMENT CHAIN: var_0 = var_1, var_1 = var_2, ..., var_{n-1} = value
    for i in range(chain_length - 1):
        tokens.extend([chain_vars[i], ASSIGN_TOKEN, chain_vars[i + 1]])
    
    # FINAL ASSIGNMENT: last_var = value
    tokens.extend([chain_vars[-1], ASSIGN_TOKEN, final_value])
    
    # QUERY: ? var_0 (asking for the value of the first variable)
    tokens.extend([QUERY_TOKEN, chain_vars[0]])
    
    # FINAL RETURN
    target = final_value
    return torch.tensor(tokens, dtype=torch.long), torch.tensor([target], dtype=torch.long)


def generate_batch(
    batch_size: int = 32,
    chain_length: int = 3, 
    seq_len: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ GENERATE A BATCH OF REASONING SAMPLES """
    
    # COLLECT SAMPLES
    inputs, targets = [], []
    
    # GENERATE SAMPLES
    for _ in range(batch_size):
        inp, tgt = generate_reasoning_sample(chain_length)
        
        # Pad to fixed sequence length
        if len(inp) < seq_len:
            padding = torch.full((seq_len - len(inp),), PAD_TOKEN, dtype=torch.long)
            inp = torch.cat([inp, padding])
        else:
            inp = inp[:seq_len]
        
        inputs.append(inp)
        targets.append(tgt)
    
    # RETURN BATCH TENSORS
    return torch.stack(inputs), torch.stack(targets).squeeze(-1)


def decode_sample(
    tokens: torch.Tensor, 
    target: int
) -> str:
    """ HUMAN-READABLE REPRESENTATION OF A SAMPLE FOR DEBUGGING """
    
    parts = []
    i = 0
    tokens_list = tokens.tolist()
    
    # DECODE TOKENS
    while i < len(tokens_list):
        tok = tokens_list[i]
        if tok == PAD_TOKEN:
            break
        elif tok == ASSIGN_TOKEN:
            parts.append('=')
        elif tok == QUERY_TOKEN:
            parts.append('?')
        elif VAR_OFFSET <= tok < VAR_OFFSET + VAR_COUNT:
            parts.append(f'v{tok - VAR_OFFSET}')
        elif 0 <= tok < VALUE_RANGE:
            parts.append(str(tok))
        else:
            parts.append(f'[{tok}]')
        i += 1
        
    # FINAL OUTPUT
    return ' '.join(parts) + f' : {target}'





# ENTRY TESTING (PRODUCTION IRRELEVANT)
if __name__ == "__main__":
    
    print("Multi-Hop Variable Tracking Dataset Generator")
    
    # TEST CHAIN LENGTHS
    for chain_len in [2, 3, 5]:
        print(f"\n Chain Length {chain_len} (Hops: {chain_len - 1}) ---")
        x, y = generate_batch(batch_size = 4, chain_length = chain_len, seq_len = 50)
        
        for i in range(min(2, len(x))):
            print(f"Sample {i + 1}: {decode_sample(x[i], y[i].item())}")
            print(f"  Raw tokens: {x[i][:15].tolist()}...")
            print(f"  Target: {y[i].item()}")
    
    print(" Data generator working")
    
