import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# FractalNet40이면 B=5, C=4


class FractalConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, dropout=None): #
        super().__init__()
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if dropout is not None:
            self.dropout = nn.Dropout2d(p=dropout, inplace=False)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.conv(x)  # conv
        if self.dropout is not None:
            out = self.dropout(out)
        out = F.relu(self.bn(out))  # -> bn -> relu
        return out


class FractalBlock(nn.Module):
    def __init__(self, n_cols, c_in, c_out, p_dropout, p_local_drop, p_global_drop, doubling=False):
        super().__init__()
        self.n_cols = n_cols
        self.cols = nn.ModuleList([nn.ModuleList() for _ in range(n_cols)])
        self.max_depth = 2**(n_cols-1)
        self.p_local_drop=p_local_drop
        self.p_global_drop=p_global_drop
        if doubling:
            self.doubler = FractalConv(c_in=c_in, c_out=c_out, kernel_size=1, padding=0)
        else:
            self.doubler = None

        n_convs = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=int)
        for col in self.cols:
            for i in range(self.max_depth):
                if (i+1)%n_convs == 0:
                    if i+1 == n_convs and not doubling:
                        cur_c_in = c_in
                    else:
                        cur_c_in = c_out
                    module = FractalConv(c_in=cur_c_in, c_out=c_out, dropout=p_dropout)
                    self.count[i] += 1
                else:
                    module = None
                col.append(module)
            n_convs //= 2

    def mask_local_drop(self, n):
        drops = np.random.binomial(1, self.p_local_drop, size=[n]).astype(bool)
        if drops.all(): # all true
            i = np.random.randint(0, n)
            drops[i]=False # Local: a join drops each input with fixed probability, but we make sure at least one survives.
        return drops


    def join(self, outs):
        if len(outs) == 1:
            return outs[0]
        if self.training: # (nn.module)
            drops = self.mask_local_drop(len(outs))
            outs = [o for drop, o in zip(drops, outs) if not drop]
        out = torch.stack(outs)
        return out.mean(dim=0)


    def forward_global(self, x, global_col):
        if self.doubler is not None:
            out = self.doubler(x)
        else:
            out = x
        n_convs = 2**(self.n_cols-1 - global_col)
        for i in range(n_convs-1, self.max_depth, n_convs):
            out = self.cols[global_col][i](out)
        return out

    def forward_local(self, x):
        if self.doubler is not None:
            out = self.doubler(x)
        else:
            out = x
        outs = [out.clone() for _ in range(self.n_cols)] # 각 col에 보낼 input
        for i in range(self.max_depth): # depth i
            st = self.n_cols - self.count[i]
            cur_outs = []
            # count[i]=4 : 0 1 2 3
            # count[i]=2 : 2 3
            for c in range(st, self.n_cols):
                cur_in = outs[c]
                cur_module = self.cols[c][i]
                cur_outs.append(cur_module(cur_in))

            # join
            joined = self.join(cur_outs)
            for c in range(st, self.n_cols):
                outs[c] = joined

        return outs[0]

    def forward(self, x, deepest=False):
        if self.training:
            # training
            if np.random.rand() < self.p_global_drop:
                global_col = np.random.randint(0, self.n_cols)
                return self.forward_global(x, global_col)
            else:
                return self.forward_local(x)
        else:
            # eval
            if deepest:
                return self.forward_global(x, self.n_cols - 1)
            else:
                return self.forward_local(x)


class fractalnet40(nn.Module):
    def __init__(self, input_size=32, num_classes=10, c_in=3, n_cols=4, channels=[64, 128, 256, 512, 512], p_dropouts=[0, 0.1, 0.2, 0.3, 0.4], p_local_drop=0.15, p_global_drop=0, doubling=False, consist_gdrop=False):
        super().__init__()
        self.B = len(channels)  # num of blocks
        self.p_dropouts = p_dropouts
        self.p_local_drop = p_local_drop
        self.p_global_drop = p_global_drop
        self.n_cols = n_cols
        self.consist_gdrop = consist_gdrop
        size = input_size

        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        c_out = c_in
        total_layers = 0
        # self, n_cols, c_in, c_out, p_dropout, p_local_drop, p_global_drop, doubling = False):
        for i in range(self.B):
            c_in, c_out = c_out, channels[i]
            fb = FractalBlock(n_cols=n_cols, c_in=c_in, c_out=c_out, p_dropout=p_dropouts[i], p_local_drop=p_local_drop, p_global_drop=p_global_drop, doubling=doubling)
            # gap?
            self.blocks.append(fb)
            self.pools.append(nn.MaxPool2d(2))
            size //= 2
            total_layers += fb.max_depth
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1] * size * size, num_classes)

    def forward(self, x, deepest=False):
        for fb, pool in zip(self.blocks, self.pools):
            x = fb(x, deepest=deepest)
            x = pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x