import torch
import numpy as np 

from diffusion_policy.dataset.normalization import get_data_stats, normalize_data


class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, obs_actions_data_file, reward_data_file):
        
        obs_action_data = np.load(obs_actions_data_file)
        self.obs_action_data = torch.tensor(normalize_data(obs_action_data, get_data_stats(obs_action_data)), dtype=torch.float32) # (30400, 36)

        self.reward_labels = torch.tensor(np.load(reward_data_file)[:, 0], dtype=torch.float32) # (30400, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.obs_action_data[idx], self.reward_labels[idx]