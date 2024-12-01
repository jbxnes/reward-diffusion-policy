import torch
import numpy as np 

from diffusion_policy.dataset.normalization import get_data_stats, normalize_data


class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, action_reward_path, train=True, reward_diff=True):
        
        # load data
        action_reward_data = np.load(action_reward_path)
        
        # separate actions and rewards
        obs_action_data = action_reward_data[:, :-2]
        reward_data = action_reward_data[:, -2:][:, int(reward_diff)]
        self.stats = get_data_stats(obs_action_data)
        
        obs_action_data = torch.tensor(normalize_data(obs_action_data, self.stats), dtype=torch.float32) # (N, 52)
        reward_data = torch.tensor(reward_data, dtype=torch.float32) # (N,)
        
        # take train or test subset 
        split_idx = int(0.8 * len(obs_action_data))
        self.obs_action_data = obs_action_data[:split_idx] if train else obs_action_data[split_idx:]
        self.reward_data = reward_data[:split_idx] if train else reward_data[split_idx:]
    
    def __len__(self):
        return len(self.obs_action_data)
    
    def __getitem__(self, idx):
        return self.obs_action_data[idx], self.reward_data[idx]