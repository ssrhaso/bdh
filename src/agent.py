import torch
import arc_agi
from arcengine import GameAction
from models.bdh import BDH, BDHConfig

class BDHAgent:
    def __init__(self, model_path=None):
        # Initialize BDH
        self.config = BDHConfig(
            vocab_size=16,  # 0-9 colors + extra for RL tokens
            n_layer=6,
            n_embd=256, 
            n_head=4, 
            mlp_internal_dim_multiplier=64
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BDH(self.config).to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            
    def get_action(self, observation):
        """
        Input: observation (Grid from ARC-AGI env)
        Output: GameAction
        """
        # 1. Flatten grid to tokens
        grid = observation['frame']  # Assuming 'frame' is the key
        tokens = torch.tensor(grid).flatten().unsqueeze(0).to(self.device)
        
        # 2. BDH Forward Pass
        # We need to map the output logits to valid actions
        # Let's say the first 8 logits correspond to ACTION1..ACTION8
        with torch.no_grad():
            logits, _ = self.model(tokens)
            
        # Take the logits from the LAST token (next-step prediction)
        last_token_logits = logits[0, -1, :8] # First 8 indices for actions
        action_idx = torch.argmax(last_token_logits).item()
        
        # Map to ARC GameAction enum (ACTION1=1, etc.)
        # This is a simplification; we need a proper action map
        return GameAction(action_idx + 1) 

    def train_step(self, obs, action, reward, next_obs):
        # This is where we will implement the RL update (PPO or simple Policy Gradient)
        pass
