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
        # (1) Neural Network Energy (학습은 Degree로 해도 괜찮음)
        img_embed = self.backbone(images)      
        act_embed = self.action_encoder(actions) 
        combined = torch.cat([img_embed, act_embed], dim=1)
        energy_nn = self.head(combined) 
        
        # (2) Kinematic Inconsistency Energy (여기가 핵심!)
        raw_actions = self.denormalize(actions)
        
        # [수정] DFK 입력 전에 Degree -> Radian 변환 필수!
        # 로봇 관절(0~7번)만 변환합니다.
        joints_deg = raw_actions[:, :8]
        joints_rad = joints_deg * (torch.pi / 180.0)  # 변환 공식 적용
        
        pred_pos_m = self.dfk(joints_rad) 
        
        # 정답(CSV)은 이미 미터(m) 단위(0.24 ~ 0.44)이므로 그대로 사용
        target_pos_m = raw_actions[:, 8:11]
        
        kinematic_error = torch.norm(pred_pos_m - target_pos_m, dim=1, keepdim=True)
        
        # Total Energy
        return energy_nn + (10.0 * kinematic_error)
    
    def compute_vision_feature(self, images):
        return self.backbone(images)

    def score_with_feature(self, img_embed, actions):
        act_embed = self.action_encoder(actions)
        combined = torch.cat([img_embed, act_embed], dim=1)
        energy_nn = self.head(combined)
        
        raw_actions = self.denormalize(actions)
        
        # [수정] 여기도 Degree -> Radian 변환 추가
        joints_deg = raw_actions[:, :8]
        joints_rad = joints_deg * (torch.pi / 180.0)
        
        pred_pos_m = self.dfk(joints_rad)
        target_pos_m = raw_actions[:, 8:11]
        
        kinematic_error = torch.norm(pred_pos_m - target_pos_m, dim=1, keepdim=True)
        
        return energy_nn + (10.0 * kinematic_error)