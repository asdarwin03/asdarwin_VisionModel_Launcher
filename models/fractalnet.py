import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def mask_local_drop(n, p):
    drops = np.random.binomial(1, p, size=[n]).astype(bool)
    if drops.all():  # all true
        i = np.random.randint(0, n)
        drops[i] = False  # Local: a join drops each input with fixed probability, but we make sure at least one survives.
    return drops


def join(xs, is_training=False, p_local_dropout=0.5):  # xs : list of input value (x) for join operation
    if len(xs) == 1:
        return xs[0]
    if is_training:  # (nn.module)
        drops = mask_local_drop(len(xs), p_local_dropout)
        xs = [o for drop, o in zip(drops, xs) if not drop]
    out = torch.stack(xs)
    return out.mean(dim=0)


class FractalConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, dropout=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if dropout is not None:
            self.dropout = nn.Dropout2d(p=dropout, inplace=False)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.conv(x)
        if self.dropout is not None:
            out = self.dropout(out)
        out = F.relu(self.bn(out))  # -> bn -> relu
        return out


class FractalBlock(nn.Module):  # x -> [y1, y2, ...]
    def __init__(self, n_cols, c_in, c_out, p_dropout, p_local_drop):
        super().__init__()
        self.path = FractalConv(c_in=c_in, c_out=c_out, dropout=p_dropout)
        self.n_cols = n_cols
        self.p_local_drop = p_local_drop
        if n_cols > 1:
            self.block1 = FractalBlock(n_cols=n_cols-1, c_in=c_in, c_out=c_out, p_dropout=p_dropout, p_local_drop=self.p_local_drop)
            self.block2 = FractalBlock(n_cols=n_cols-1, c_in=c_out, c_out=c_out, p_dropout=p_dropout, p_local_drop=self.p_local_drop)
        else:
            self.block1 = None
            self.block2 = None

    def forward(self, x, only_col=None):  # 어떤 리스트를 반환해야 함
        # 1. FractalBlock에 처음 들어올때의 입력은 x를 (이전 결과를 join했거나 어쨌든 1개) 리스트
        # forward 결과는 각 path를 통한 리스트
        # 별로 join 연산은 따로 해야함
        ys = []
        if only_col is None:
            if self.n_cols > 1:
                ys = self.block2(join(xs=self.block1(x, only_col=only_col), is_training=self.training, p_local_dropout=self.p_local_drop), only_col=only_col)
            ys.append(self.path(x))
        elif only_col == self.n_cols:  # fixed global path
            ys.append(self.path(x))
        else:  # only_col and only_col < n_cols
            ys = self.block2(x=join(xs=self.block1(x, only_col=only_col), is_training=self.training, p_local_dropout=self.p_local_drop), only_col=only_col)
        return ys


class fractalnet(nn.Module):
    def __init__(self, input_size=32, num_classes=10, c_in=3, n_cols=4, channels=[64, 128, 256, 512, 512], p_dropouts=[0, 0.1, 0.2, 0.3, 0.4], p_local_drop=0.15, p_global_drop=0):
        super().__init__()
        self.B = len(channels)  # num of blocks
        self.p_dropouts = p_dropouts
        self.p_local_drop = p_local_drop
        self.p_global_drop = p_global_drop
        self.n_cols = n_cols
        size = input_size

        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        c_out = c_in
        # self, n_cols, c_in, c_out, p_dropout, p_local_drop, p_global_drop):
        for i in range(self.B):
            c_in, c_out = c_out, channels[i]
            fb = FractalBlock(n_cols=n_cols, c_in=c_in, c_out=c_out, p_dropout=p_dropouts[i], p_local_drop=p_local_drop)
            self.blocks.append(fb)
            self.pools.append(nn.MaxPool2d(2))
            size //= 2
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1] * size * size, num_classes)

    def forward(self, x):
        if np.random.rand() < self.p_global_drop:
            global_col = np.random.randint(1, self.n_cols+1)
        else:
            global_col = None

        for fb, pool in zip(self.blocks, self.pools):
            xs = fb(x, only_col=global_col)
            x = join(xs=xs, is_training=self.training, p_local_dropout=self.p_local_drop)
            x = pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x