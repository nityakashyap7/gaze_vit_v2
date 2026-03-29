import torch
import numpy as np
from utils.hook import AttentionExtractor
from utils.logger import Logger
from GABRIL_utils import utils
from GABRIL_utils.atari_env_manager import create_env as create_atari_environment


class RolloutRunner:

    def __init__(self, model, envs, device):
        self.model = model
        self.envs = envs
        self.device = device
    
    def run(self, n_episodes = 1000):
        
        self.model.eval() 
        total_rewards = []
        for env in self.envs:
            rewards = [0 for i in range(n_episodes)]
            with torch.no_grad():
                for episode in range(n_episodes):
                    obs, _ = env.reset()
                    done = False

                    while not done:
                        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)/255.0 # add batch dim

                        
                        action_pred = self.model(obs_tensor)

                        action = torch.argmax(action_pred, dim=-1).item() # get the predicted action index
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        rewards[episode] += reward
                         
            avg_reward = np.mean(rewards) 
            total_rewards.append(avg_reward)
        
        return total_rewards

        
                

