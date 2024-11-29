import torch.nn as nn 


class RewardModel(nn.Module):
    xdef __init__(self):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(36, 16),
                                 nn.SiLU(),
                                 nn.Linear(16, 16),
                                 nn.SiLU(),
                                 nn.Linear(16, 16),
                                 nn.SiLU(),
                                 nn.Linear(16, 16),
                                 nn.SiLU(),
                                 nn.Linear(16, 1))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.net(x)
        x = self.tanh(x)
        return x