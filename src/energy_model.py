import torch
import torch.nn as nn
try:
    from model_architecture import VisionEncoder
except ImportError:
    from model_architecture import VisionEncoder
try:
    from dfk_layer import DifferentiableFK
except ImportError:
    from dfk_layer import DifferentiableFK

class EnergyModel(nn.Module):
    def __init__(self, action_dim=14, stats=None, device='cpu'):
        super().__init__()
        
        # 1. Vision Encoder
        self.encoder = VisionEncoder()
        
        # Modify first layer for 6-channel (Stereo) input
        original_conv = self.encoder.features[0]
        self.encoder.features[0] = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # 2. Energy MLP
        input_dim = self.encoder.output_dim + action_dim
        
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        # Kinematic Layer
        self.dfk = DifferentiableFK(device=device)
        
        # Register Action Stats for Denormalization
        if stats is not None:
            self.register_buffer('action_min', torch.tensor(stats['min'], dtype=torch.float32))
            self.register_buffer('action_max', torch.tensor(stats['max'], dtype=torch.float32))
        else:
            self.register_buffer('action_min', torch.zeros(action_dim))
            self.register_buffer('action_max', torch.ones(action_dim))

    def forward(self, img, action):
        visual_emb = self.encoder(img)
        x = torch.cat([visual_emb, action], dim=1)
        energy_nn = self.energy_net(x)
        
        # Calculate Kinematic Consistency Loss during Training
        raw_actions = self.denormalize(action)
        
        # Convert Degree to Radian for DFK (Critical for correct calculation)
        joints_deg = raw_actions[:, :6] 
        joints_rad = joints_deg * (torch.pi / 180.0)
        
        # Compute DFK with TCP offset and calculate error
        pred_pos = self.dfk(joints_rad)
        target_pos = raw_actions[:, 8:11]
        
        k_err = torch.norm(pred_pos - target_pos, dim=1, keepdim=True)

        # Final Energy = NN Energy + Weighted Kinematic Error
        return energy_nn + (10.0 * k_err)

    def compute_vision_feature(self, img):
        return self.encoder(img)

    def score_with_feature(self, vision_feature, action):
        x = torch.cat([vision_feature, action], dim=1)
        energy_nn = self.energy_net(x)
        
        # Kinematic Consistency Check during Inference
        raw_actions = self.denormalize(action)
        
        # Convert Degree to Radian for DFK
        joints_deg = raw_actions[:, :6]
        joints_rad = joints_deg * (torch.pi / 180.0)
        
        pred_pos = self.dfk(joints_rad)
        target_pos = raw_actions[:, 8:11]
        
        k_err = torch.norm(pred_pos - target_pos, dim=1, keepdim=True)
        return energy_nn + (10.0 * k_err)

    def denormalize(self, norm_action):
        # Maps actions from [-1, 1] back to original scale.
        return (norm_action + 1) / 2 * (self.action_max - self.action_min) + self.action_min