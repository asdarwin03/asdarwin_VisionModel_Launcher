import torch
from torch import nn
import torch.nn.functional as F
from method import Method

class RotNet(Method):
    def __init__(self, encoder):
        super().__init__(encoder=encoder)
        self.rotation_head = self.make_head(self.encoder.dim_out, 4) # 0, 90, 180, 270
    
    def forward(self, batch):
        x, y = batch
        rotated_x, rotated_y = self.make_rotated_batch(x)
        rotated_x = self.encoder(rotated_x)
        y_pred = self.rotation_head(rotated_x)
        return F.cross_entropy(y_pred, rotated_y)
    
    def make_head(self, feature_dim, num_classes):
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def make_rotated_batch(self, x):
        rotations = [torch.rot90(x, k=k, dims=(2, 3)) for k in range(4)]
        rotated = torch.cat(rotations, dim=0)
        y = torch.arange(4, device=x.device).repeat_interleave(x.size(0))
        return rotated, y
