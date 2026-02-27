import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outputs, outputs, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class resnet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            # stride=1
            # 16 channel로 온다고 가정 (conv1 outchannel), 3 block
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
        )

        self.layer12_downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32)
        )

        self.layer2 = nn.Sequential(
            # stride=2
            # 16 channel inputs -> 32 channel outputs
            # 32
            ResidualBlock(16, 32, 2, self.layer12_downsample),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )

        self.layer23_downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )

        self.layer3 = nn.Sequential(
            # stride=2
            # 32 channel inputs -> 64 channel outputs
            ResidualBlock(32, 64, 2, self.layer23_downsample),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x