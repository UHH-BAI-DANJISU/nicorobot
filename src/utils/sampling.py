import torch

def sample_negative_actions(action, K=64, std=0.1):
    B, D = action.shape
    noise = torch.randn(B, K, D, device=action.device) * std
    return action.unsqueeze(1) + noise
