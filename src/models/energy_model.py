import torch
import torch.nn as nn
from .vision import VisionEncoder

class EnergyModel(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = VisionEncoder()
        self.energy = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, action):
        z = self.encoder(obs)
        x = torch.cat([z, action], dim=1)
        return self.energy(x)
