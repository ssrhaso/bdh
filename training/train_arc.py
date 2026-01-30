import arc_agi
from src.agent import BDHAgent

def train():
    # 1. Setup Environment
    arc = arc_agi.Arcade()
    # 'ls20' is the example game from the docs
    env = arc.make("ls20", render_mode="terminal") 
    
    # 2. Setup Agent
    agent = BDHAgent()
    
    print("Starting RL Training Loop on ARC-AGI-3...")
    
    for episode in range(10):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Agent decides
            action = agent.get_action(obs)
            
            # Env steps
            # Note: The docs say env.step(action)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            # (Optional) Store transition for training
            # agent.memory.push(obs, action, reward, next_obs)
            
        print(f"Episode {episode} finished. Score: {total_reward}")
        
    print(arc.get_scorecard())

if __name__ == "__main__":
    train()
