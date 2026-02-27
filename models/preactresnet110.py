import torch
from torch import nn
import torch.nn.functional as F

# n = 18 -> (108 + 2 layers)
class PreActBasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.downsample:
            x = self.downsample(x)
        out += x
        return out


class preactresnet110(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer12_downsample = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),) # nn.BatchNorm2d(32) 없앰
        self.layer23_downsample = nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),) # nn.BatchNorm2d(64) 없앰
        self.bn = nn.BatchNorm2d(64)

        self.layer1_list = []
        self.layer2_list = [PreActBasicBlock(16, 32, 2, self.layer12_downsample)]
        self.layer3_list = [PreActBasicBlock(32, 64, 2, self.layer23_downsample)]

        for i in range(0, 18):
            self.layer1_list.append(PreActBasicBlock(16, 16))

        for i in range(0, 17):
            self.layer2_list.append(PreActBasicBlock(32, 32))
            self.layer3_list.append(PreActBasicBlock(64, 64))

        self.layer1 = nn.Sequential(*self.layer1_list)
        self.layer2 = nn.Sequential(*self.layer2_list)
        self.layer3 = nn.Sequential(*self.layer3_list)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)


    def forward(self, x):
        x = self.conv1(x)  # preact
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x