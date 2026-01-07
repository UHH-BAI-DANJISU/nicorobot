# baseline code

import torch.nn as nn
from .vision import VisionEncoder

class ExplicitBC(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = VisionEncoder()
        self.policy = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs):
        z = self.encoder(obs)
        return self.policy(z)
