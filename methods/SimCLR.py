import torch
from torch import nn
from torch.nn import functional as F
from method import Method

class SimCLR(Method):
    def __init__(self, encoder, z_dim=128, temperature=0.1):
        super().__init__(encoder=encoder)
        self.temperature = temperature
        self.z_dim = z_dim
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.dim_out, self.encoder.dim_out),
            nn.ReLU(),
            nn.Linear(self.encoder.dim_out, self.z_dim))

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=0)
        z = self.projector(self.encoder(x))
        z = F.normalize(z, dim=1)

        logits = torch.mm(z, z.T) / self.temperature
        labels = torch.arange(x1.size(0), 2 * x1.size(0)).to(x.device)
        labels = torch.cat([labels, torch.arange(0, x1.size(0)).to(x.device)])
        logits.fill_diagonal_(float('-inf'))
        
        return F.cross_entropy(logits, labels)