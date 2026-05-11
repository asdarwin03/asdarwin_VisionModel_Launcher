import torch
from torch import nn


class SimCLR(nn.Module):
    def __init__(self, encoder, feature_dim=256, net_config=None):
        super().__init__()
        if net_config is not None:
            feature_dim = net_config['feature_dim']
        self.encoder = encoder # (c_in, feature_dim)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z
