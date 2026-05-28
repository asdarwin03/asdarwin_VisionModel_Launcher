from torch import nn

class Encoder(nn.Module):
    def __init__(self, dim_out=256):
        super().__init__()
        self.dim_out = dim_out

    def forward(self, x):
        # x: (B, C, H, W) -> (B, dim_out)
        pass