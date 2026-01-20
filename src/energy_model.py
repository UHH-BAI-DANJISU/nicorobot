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
        
        # 6채널 입력 (Stereo)
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
        
        self.dfk = DifferentiableFK(device=device)
        
        # Stats 등록
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
        
        # [수정] Training 중 DFK Loss 계산
        raw_actions = self.denormalize(action)
        
        # >>> [핵심 복구] 데이터가 Degree이므로 Radian으로 변환 필수! <<<
        joints_deg = raw_actions[:, :6] 
        joints_rad = joints_deg * (torch.pi / 180.0)
        
        # DFK (이제 오프셋 보정됨) & Error
        pred_pos = self.dfk(joints_rad)
        target_pos = raw_actions[:, 8:11]
        
        k_err = torch.norm(pred_pos - target_pos, dim=1, keepdim=True)
        return energy_nn + (10.0 * k_err)

    def compute_vision_feature(self, img):
        return self.encoder(img)

    def score_with_feature(self, vision_feature, action):
        x = torch.cat([vision_feature, action], dim=1)
        energy_nn = self.energy_net(x)
        
        # [수정] Inference 중 DFK Consistency Check
        raw_actions = self.denormalize(action)
        
        # >>> [핵심 복구] Inference에서도 변환 필수! <<<
        joints_deg = raw_actions[:, :6]
        joints_rad = joints_deg * (torch.pi / 180.0)
        
        pred_pos = self.dfk(joints_rad)
        target_pos = raw_actions[:, 8:11]
        
        k_err = torch.norm(pred_pos - target_pos, dim=1, keepdim=True)
        return energy_nn + (10.0 * k_err)

    def denormalize(self, norm_action):
        return (norm_action + 1) / 2 * (self.action_max - self.action_min) + self.action_min