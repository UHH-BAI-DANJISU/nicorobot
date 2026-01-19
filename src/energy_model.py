import torch
import torch.nn as nn
from torchvision.models import resnet18
try:
    from models.dfk_layer import DifferentiableFK
except ImportError:
    from dfk_layer import DifferentiableFK

class EnergyModel(nn.Module):
    def __init__(self, action_dim=14, stats=None, device='cpu'):
        super().__init__()
        # 1. Vision Encoder (6채널 설정)
        self.backbone = resnet18(pretrained=True)
        old_weight = self.backbone.conv1.weight
        new_weight = torch.cat([old_weight, old_weight], dim=1) 
        self.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(new_weight)
        self.backbone.fc = nn.Identity() 
        
        # 2. Action Encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2)
        )
        
        # 3. Energy Head
        self.head = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        self.dfk = DifferentiableFK(device=device)
        
        if stats is not None:
            min_val = torch.tensor(stats['min'], dtype=torch.float32) if not isinstance(stats['min'], torch.Tensor) else stats['min']
            max_val = torch.tensor(stats['max'], dtype=torch.float32) if not isinstance(stats['max'], torch.Tensor) else stats['max']
            self.register_buffer('action_min', min_val)
            self.register_buffer('action_max', max_val)

    def denormalize(self, norm_action):
        return (norm_action + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    def forward(self, images, actions):
        # [수정] 모든 항이 '낮을수록 정답'이 되도록 구성
        img_embed = self.backbone(images)      
        act_embed = self.action_encoder(actions) 
        combined = torch.cat([img_embed, act_embed], dim=1)
        energy_nn = self.head(combined) 
        
        raw_actions = self.denormalize(actions)
        pred_pos_m = self.dfk(raw_actions[:, :8]) 
        kinematic_error = torch.norm(pred_pos_m - raw_actions[:, 8:11], dim=1, keepdim=True)
        
        # 신경망 출력(energy_nn) + 물리 오차 페널티(kinematic_error)
        return energy_nn + (10.0 * kinematic_error)
    
    def compute_vision_feature(self, images):
        return self.backbone(images)

    def score_with_feature(self, img_embed, actions):
        # [수정] 추론 시에도 학습과 동일한 에너지 계산식 적용
        act_embed = self.action_encoder(actions)
        combined = torch.cat([img_embed, act_embed], dim=1)
        energy_nn = self.head(combined)
        
        raw_actions = self.denormalize(actions)
        pred_pos_m = self.dfk(raw_actions[:, :8])
        kinematic_error = torch.norm(pred_pos_m - raw_actions[:, 8:11], dim=1, keepdim=True)
        
        return energy_nn + (10.0 * kinematic_error)