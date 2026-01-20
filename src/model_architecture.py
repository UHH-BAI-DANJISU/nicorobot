import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer: Converts feature maps (C x H x W) into feature points (C x 2).
    Draft Reference: "The output feature map is flattened and passed through a spatial softmax layer" 
    """
    def __init__(self, temperature=None, normalize=False):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, feature_map):
        # feature_map: [Batch, C, H, W]
        batch, c, h, w = feature_map.shape
        
        # Create coordinate grid
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feature_map.device),
            torch.linspace(-1, 1, w, device=feature_map.device)
        )
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        
        # Flatten map: [Batch, C, H*W]
        flat_map = feature_map.view(batch, c, -1)
        
        if self.temperature:
            flat_map = flat_map / self.temperature
            
        softmax_attention = F.softmax(flat_map, dim=2) # [Batch, C, H*W]
        
        # Calculate expected coordinates
        expected_x = torch.sum(pos_x * softmax_attention, dim=2, keepdim=True)
        expected_y = torch.sum(pos_y * softmax_attention, dim=2, keepdim=True)
        
        # Output: [Batch, C*2]
        expected_coords = torch.cat([expected_x, expected_y], dim=2).view(batch, -1)
        return expected_coords

class VisionEncoder(nn.Module):
    """
    Identical Encoder for both Implicit and Explicit Policies.
    Backbone: ResNet-18 
    Pooling: Spatial Softmax 
    """
    def __init__(self, input_shape=(3, 64, 64)):
        super().__init__()
        # Load ResNet18
        resnet = resnet18(pretrained=True)
        
        # Remove average pooling and fc layer
        # Note: For 64x64 input, standard ResNet18 downsamples to 2x2 at layer4.
        # We want slightly larger maps for Spatial Softmax, so we might cut it earlier 
        # OR accept 2x2. Here we use layers up to layer4.
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4 
        )
        
        # Calculate output dim for Spatial Softmax
        # ResNet18 layer4 output channels = 512.
        # Spatial Softmax output dim = Channels * 2 (x,y)
        self.spatial_softmax = SpatialSoftmax()
        self.output_dim = 512 * 2  # 1024 features

    def forward(self, x):
        x = self.features(x)       # [B, 512, 2, 2] given 64x64 input
        z = self.spatial_softmax(x) # [B, 1024]
        return z