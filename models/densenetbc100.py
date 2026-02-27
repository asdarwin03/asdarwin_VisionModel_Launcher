import math

import torch
from torch import nn
import torch.nn.functional as F

# init lr = 0.1, /10 at 50% and 75% of the total number of training epochs
# batchsize=64 for 300 and 40 epochs
# weight_decay = 10e-4 = 0.0001
# momentum : Nesterov momentum 0.9 without dampening
# CIFAR10을 위한 densenet에는
# 모든 블럭에 대해 동일한 DenseLayer 개수가 들어감
# Dense Block은 총 3개
# Transition layers 영역이 2개
# softmax가 1개, 첫 conv 1개 해서
# 100-4 = 96
# 96/3 = 32
# DenseBlock 당 32/2=16개의 DenseLayer가 들어가면 DenseNet-BC, Depth=100

class DenseLayer(nn.Module):
    # Bottleneck
    def __init__(self, c_in, k): #
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=4*k, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(4*k)
        self.conv2 = nn.Conv2d(in_channels=4*k, out_channels=k, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))  # bn -> relu -> 1x1 conv
        out = self.conv2(F.relu(self.bn2(out)))  # bn -> relu -> 3x3 conv
        return torch.cat([x, out], dim=1)  # concatenate operation. number of output feature map : k is concatenated

class DenseBlock(nn.Module):
    def __init__(self, k0, num_layers, k):
        super().__init__()

        denselayer_list=[]
        cumulated_k = k0 # cin
        for i in range(0, num_layers):
            denselayer_list.append(DenseLayer(cumulated_k, k))
            cumulated_k += k
        self.dense_layers = nn.Sequential(*denselayer_list)

    def forward(self, x):
        return self.dense_layers(x)

class TransitionLayer(nn.Module):
    def __init__(self, m, theta):
        super().__init__()
        self.bn = nn.BatchNorm2d(m)
        self.conv1 = nn.Conv2d(in_channels=m, out_channels=math.floor(theta*m), kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(F.relu(self.bn(x)))
        x = self.avgpool(x)
        return x

class densenetbc100(nn.Module):
    def __init__(self, k=12, total_layers=100, theta=0.5, num_classes=10):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=2*k, kernel_size=3, stride=1, padding=1, bias=False)
        num_layers = (total_layers-4)//6
        k0=2*k

        self.denseblock1 = DenseBlock(k0=k0, num_layers=num_layers, k=k) # 24, 16, 12
        m = k0 + num_layers*k
        self.translayer1 = TransitionLayer(m=m, theta=theta)
        k0 = math.floor(theta * m)

        self.denseblock2 = DenseBlock(k0=k0, num_layers=num_layers, k=k)
        m = k0 + num_layers*k
        self.translayer2 = TransitionLayer(m=m, theta=theta)
        k0 = math.floor(theta * m)

        self.denseblock3 = DenseBlock(k0=k0, num_layers=num_layers, k=k)
        m = k0 + num_layers*k
        self.bn = nn.BatchNorm2d(m)
        self.globalavgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(m, num_classes)



    def forward(self, x):
        x = self.first_conv(x)

        x = self.denseblock1(x)
        x = self.translayer1(x)

        x = self.denseblock2(x)
        x = self.translayer2(x)

        x = self.denseblock3(x)

        x = F.relu(self.bn(x))
        x = self.globalavgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x