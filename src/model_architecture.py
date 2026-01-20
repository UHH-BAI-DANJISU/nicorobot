import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=None):
        super().__init__()
        self.temperature = temperature

    def forward(self, feature_map):
        batch, c, h, w = feature_map.shape
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feature_map.device),
            torch.linspace(-1, 1, w, device=feature_map.device),
            indexing='ij'
        )
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        
        flat_map = feature_map.view(batch, c, -1)
        if self.temperature:
            flat_map = flat_map / self.temperature
            
        softmax_attention = F.softmax(flat_map, dim=2)
        expected_x = torch.sum(pos_x * softmax_attention, dim=2, keepdim=True)
        expected_y = torch.sum(pos_y * softmax_attention, dim=2, keepdim=True)
        
        # [Batch, C*2]
        return torch.cat([expected_x, expected_y], dim=2).view(batch, -1)

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        # Layer4까지만 사용 (Spatial Softmax를 위해)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.spatial_softmax = SpatialSoftmax()
        # ResNet18 Layer4(512ch) -> 512*2(x,y) = 1024
        self.output_dim = 1024 

    def forward(self, x):
        x = self.features(x)
        z = self.spatial_softmax(x)
        return z