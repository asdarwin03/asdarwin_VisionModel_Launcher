import torch
from torch import nn


class MixerLayer(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim, padding="same")
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim)

        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        y = self.bn1(self.gelu1(self.conv1(x)))
        x = x + y
        x = self.bn2(self.gelu2(self.conv2(x)))
        return x


class convmixer(nn.Module):
    def __init__(self, num_classes=10, c_in=3, dim=256, depth=12, patch_size=7, kernel_size=9):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=c_in, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)
        self.mixerlayers = nn.ModuleList([
            MixerLayer(dim=dim, kernel_size=kernel_size) for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=dim, out_features=num_classes)

    def forward(self, x: torch.Tensor):
        # x : (B, C, H, W)
        x = self.bn(self.gelu(self.conv(x)))  # (B, D, H/P, W/P)
        for mixerlayer in self.mixerlayers:
            x = mixerlayer(x)
        x = self.fc(self.flatten(self.pool(x)))
        return x
