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
        # 1. Vision Encoder
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
        """
        [훈련용] 물리 오차(kinematic_error)를 제거!
        모델이 오직 '이미지'와 '행동'의 관계만 보고 학습하도록 강제합니다.
        """
        img_embed = self.backbone(images)      
        act_embed = self.action_encoder(actions) 
        combined = torch.cat([img_embed, act_embed], dim=1)
        energy_nn = self.head(combined) 
        
        # 물리 오차 항 삭제 (꼼수 방지)
        return energy_nn 

    def compute_vision_feature(self, images):
        return self.backbone(images)

    def score_with_feature(self, img_embed, actions):
        """
        [추론용] 실전에서는 물리 오차를 켜서, 로봇이 불가능한 자세를 취하지 않도록 가이드합니다.
        """
        act_embed = self.action_encoder(actions)
        combined = torch.cat([img_embed, act_embed], dim=1)
        energy_nn = self.head(combined)
        
        # 추론 시: 신경망 에너지 + 물리 가이드
        return energy_nn