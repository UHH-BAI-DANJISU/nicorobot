import torch
import torch.nn as nn
try:
    from model_architecture import VisionEncoder
except ImportError:
    # inference.py 실행 시 경로 문제 방지
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model_architecture import VisionEncoder

class EnergyModel(nn.Module):
    def __init__(self, action_dim=14, stats=None, device='cpu'):
        super().__init__()
        
        # ------------------------------------------------------------------
        # [구조 일치] train_implicit.py와 동일한 Vision Encoder 사용
        # ------------------------------------------------------------------
        self.encoder = VisionEncoder() 
        
        # [중요] 6채널 입력 (Stereo) 수정 로직도 동일하게 적용
        original_conv = self.encoder.features[0]
        self.encoder.features[0] = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # ------------------------------------------------------------------
        # [구조 일치] train_implicit.py와 동일한 MLP (키 이름: energy_net)
        # ------------------------------------------------------------------
        input_dim = self.encoder.output_dim + action_dim
        
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Min-Max Stats 저장 (Denormalization용)
        if stats is not None:
            self.register_buffer('action_min', torch.tensor(stats['min'], dtype=torch.float32))
            self.register_buffer('action_max', torch.tensor(stats['max'], dtype=torch.float32))
        else:
            # 로드 시 에러 방지를 위해 임시 버퍼 등록 (나중에 덮어씌워짐)
            self.register_buffer('action_min', torch.zeros(action_dim))
            self.register_buffer('action_max', torch.ones(action_dim))

    def forward(self, img, action):
        visual_emb = self.encoder(img)
        x = torch.cat([visual_emb, action], dim=1)
        return self.energy_net(x)

    def compute_vision_feature(self, img):
        """Inference 최적화용: 이미지 특징만 추출"""
        return self.encoder(img) # [B, 1024]

    def score_with_feature(self, vision_feature, action):
        """Inference 최적화용: 특징+액션 -> 에너지 점수 (Pure Neural Energy)"""
        x = torch.cat([vision_feature, action], dim=1)
        return self.energy_net(x)

    def denormalize(self, norm_action):
        """[-1, 1] -> [Min, Max]"""
        return (norm_action + 1) / 2 * (self.action_max - self.action_min) + self.action_min