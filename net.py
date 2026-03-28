import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class YawRegressor(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),  # (sin_yaw, cos_yaw)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return F.normalize(out, dim=1)  # 单位圆上的向量
