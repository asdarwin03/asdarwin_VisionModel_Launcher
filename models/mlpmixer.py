import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, d_in, hidden_d):
        super().__init__()
        self.d_in = d_in
        self.hidden_d = hidden_d
        self.dense1 = nn.Linear(in_features=self.d_in, out_features=self.hidden_d)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(in_features=self.hidden_d, out_features=self.d_in)

    def forward(self, x: torch.Tensor):
        # x : (B, a, b) (transposed)
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        # x : (B, a, b)
        return x


class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, patches_dim, channels_dim):
        # patches_dim = D_S, channels_dim = D_C
        super().__init__()
        self.num_patches = num_patches
        self.C = hidden_size

        self.ln1 = nn.LayerNorm(hidden_size)
        self.mlp1 = MLPBlock(d_in=num_patches, hidden_d=patches_dim)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp2 = MLPBlock(d_in=hidden_size, hidden_d=channels_dim)

    def forward(self, x):
        # x : (B, num_patches, channels)
        y = self.ln1(x)
        y = y.transpose(1, 2)  # -> (B, channels, num_patches)
        y = self.mlp1(y)
        y = y.transpose(1, 2)  # -> (B, num_patches, channels)
        x = x + y  # skip connection

        # x : (B, num_patches, channels)
        y = self.ln2(x)
        y = self.mlp2(y)
        x = x + y  # skip connection
        return x


class mlpmixer(nn.Module):
    def __init__(self, num_classes=10, img_size=(224, 224), num_mixerlayers=8, patch_size=16, C=512, c_in=3):
        super().__init__()
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.P = patch_size  # patch resolution
        self.N = num_mixerlayers  # number of layers
        self.C = C  # hidden size

        self.fcperpatch = nn.Linear(in_features=c_in * (self.P ** 2), out_features=self.C)  # 마지막 차원에 대해 fc
        self.mixerlayers = nn.ModuleList([
            MixerLayer(num_patches=self.num_patches, hidden_size=self.C, patches_dim=self.C // 2, channels_dim=4 * self.C) for _ in range(self.N)
        ])

        self.ln = nn.LayerNorm(self.C)
        self.fc = nn.Linear(self.C, num_classes)

    def forward(self, x: torch.Tensor):
        # x : (B, C, H, W)
        batch_size = x.size(0)
        patches = x.unfold(2, self.P, self.P).unfold(3, self.P, self.P)  # unfold(dim, size, stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, C, H/P, W/P, P, P) -> (B, H/P, W/P, C, P, P)
        patches = patches.flatten(start_dim=3)  # -> (B, H/P, W/P, CPP)
        patches = patches.flatten(1, 2)  # -> (B, num_patches=HW/PP, CPP)

        x = self.fcperpatch(patches)  # -> (B, num_patches, channels(C))
        for mixerlayer in self.mixerlayers:
            x = mixerlayer(x)  # -> (B, num_patches, channels(C))
        x = self.ln(x)  # -> (B, num_patches, LayerNormed_channels)
        x = x.mean(dim=1)  # -> (B, channels(C))
        x = self.fc(x)  # -> (B, num_classes)
        return x
