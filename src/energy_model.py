import torch
import torch.nn as nn
from torchvision.models import resnet18
# 만약 dfk_layer가 models 폴더 안에 있다면:
# from models.dfk_layer import DifferentiableFK 
# 혹은 같은 폴더에 있다면:
try:
    from models.dfk_layer import DifferentiableFK
except ImportError:
    from dfk_layer import DifferentiableFK

class EnergyModel(nn.Module):
    def __init__(self, action_dim=14, stats=None, device='cpu'):
        """
        args:
            action_dim: 14 (Joint 8 + Cartesian 6)
            stats: 데이터셋 정규화 통계 {'min': ..., 'max': ...}
            device: 'cpu' 또는 'cuda' (DFK로 전달됨) [추가됨!]
        """
        super().__init__()
        
        # 1. Vision Encoder (ResNet18)
        # weights='IMAGENET1K_V1' 권장되지만, 호환성을 위해 pretrained=True 유지 가능
        self.backbone = resnet18(pretrained=True)
        
        # 입력 채널 수정: 3채널 -> 6채널 (Left + Right Eye)
        # 기존 가중치를 복사해서 초기화 (학습 가속화)
        old_weight = self.backbone.conv1.weight
        new_weight = torch.cat([old_weight, old_weight], dim=1) # [64, 6, 7, 7]
        
        self.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(new_weight)

        self.backbone.fc = nn.Identity() # 512차원 특징 벡터 추출
        
        # 2. Action Encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2)
        )
        
        # 3. Energy Head (Vision + Action -> Energy Scalar)
        self.head = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        # 4. [수정됨] DFK에 device 전달
        self.dfk = DifferentiableFK(device=device)
        
        # 5. [수정됨] 통계값 등록 (Numpy -> Tensor 자동 변환 추가)
        if stats is not None:
            # 안전장치: numpy array가 들어오면 tensor로 변환
            min_val = stats['min']
            max_val = stats['max']
            
            if not isinstance(min_val, torch.Tensor):
                min_val = torch.tensor(min_val, dtype=torch.float32)
            if not isinstance(max_val, torch.Tensor):
                max_val = torch.tensor(max_val, dtype=torch.float32)
            
            self.register_buffer('action_min', min_val)
            self.register_buffer('action_max', max_val)
        else:
            print("[Warning] Stats가 제공되지 않았습니다.")
            self.action_min = None
            self.action_max = None

    def denormalize(self, norm_action):
        """ -1~1 사이의 정규화된 Action을 원래 스케일로 복원 """
        if self.action_min is None:
            return norm_action
        
        # 버퍼에 있는 값 사용 (자동으로 같은 device에 있음)
        return (norm_action + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    def forward(self, images, actions):
        """
        images: [B, 6, 64, 64]
        actions: [B, 14] (Normalized)
        """
        # (1) Neural Network Energy
        img_embed = self.backbone(images)      
        act_embed = self.action_encoder(actions) 
        
        combined = torch.cat([img_embed, act_embed], dim=1)
        energy_nn = self.head(combined) 
        
        # (2) Kinematic Inconsistency Energy
        raw_actions = self.denormalize(actions)
        
        # 데이터셋 구조: 0~7(Joint), 8~10(Pos)
        joints_rad = raw_actions[:, :8] 
        target_pos_m = raw_actions[:, 8:11]
        
        # DFK 계산
        pred_pos_m = self.dfk(joints_rad) 
        
        # 오차 계산
        kinematic_error = torch.norm(pred_pos_m - target_pos_m, dim=1, keepdim=True)
        
        # Total Energy (Lambda = 10.0)
        total_energy = energy_nn + (10.0 * kinematic_error)
        
        return total_energy