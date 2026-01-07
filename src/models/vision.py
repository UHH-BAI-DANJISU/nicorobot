import torch.nn as nn
from torchvision.models import resnet18

class VisionEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        resnet = resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
