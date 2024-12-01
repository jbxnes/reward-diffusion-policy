import torch
import numpy as np 

from diffusion_policy.dataset.normalization import get_data_stats, normalize_data


class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, obs_actions_path, rewards_path, train=True, reward_diff=True):
        
        # load data
        obs_action_data = np.load(obs_actions_path)
        self.stats = get_data_stats(obs_action_data)
        
        obs_action_data = torch.tensor(normalize_data(obs_action_data, self.stats), dtype=torch.float32) # (N, 36)
        reward_data = torch.tensor(np.load(rewards_path)[:, int(reward_diff)], dtype=torch.float32) # (N,)

        assert obs_action_data.shape[0] == reward_data.shape[0], 'obs_action_data and reward_data must have the same length'        
        
        # split into train or test
        split_idx = int(0.8 * len(obs_action_data))
        self.obs_action_data = obs_action_data[:split_idx] if train else obs_action_data[split_idx:]
        self.reward_data = reward_data[:split_idx] if train else reward_data[split_idx:]
    
    def __len__(self):
        return len(self.obs_action_data)
    
    def __getitem__(self, idx):
        return self.obs_action_data[idx], self.reward_data[idx]