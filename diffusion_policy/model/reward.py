import torch.nn as nn 


class RewardModel(nn.Module):
    def __init__(self, reward_diff=True):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(52, 16),
                                 nn.SiLU(),
                                 nn.Linear(16, 16),
                                 nn.SiLU(),
                                 nn.Linear(16, 16),
                                 nn.SiLU(),
                                 nn.Linear(16, 16),
                                 nn.SiLU(),
                                 nn.Linear(16, 1))
        
        if reward_diff:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.activation(x)
        return x.squeeze()