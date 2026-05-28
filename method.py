from torch import nn

class Method(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self):
        pass

    def forward_features(self, batch):
        x, _ = batch
        return self.encoder(x)