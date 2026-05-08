import torch
from torch import nn


class RotNet(nn.Module):
    def __init__(self, encoder, num_classes=10, c_in=3, feature_dim=256, net_config=None):
        super().__init__()
        if net_config is not None:
            c_in = net_config['c_in']
            feature_dim = net_config['feature_dim']
            num_classes = net_config['head_config']['num_classes']

        self.encoder = encoder
        self.classifier = self.make_head(feature_dim, num_classes)
        self.rotation_head = self.make_head(feature_dim, 4)
        self.mode = "pretrain"
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        if net_config is not None:
            if net_config['head_config']['finetune_type'] == "Linear":
                self.classifier = nn.Linear(feature_dim, num_classes)
            elif net_config['head_config']['finetune_type'] == "MLP":
                self.classifier = self.make_head(feature_dim, num_classes)
    
    def make_head(self, feature_dim, num_classes):
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )
    
    def setmode(self, mode):
        self.mode = mode
    

    def make_rotated_batch(self, x):
        rotations = [torch.rot90(x, k=k, dims=(2, 3)) for k in range(4)]
        rotated = torch.cat(rotations, dim=0)
        y = torch.arange(4, device=x.device).repeat_interleave(x.size(0))
        return rotated, y

    def forward(self, x):
        if self.mode == "pretrain":
            return self.forward_pretrain(x)
        else:
            return self.forward_finetune(x)

    def forward_pretrain(self, x):
        x, y = self.make_rotated_batch(x)
        x = self.pool(self.encoder(x))
        x = self.flatten(x)
        x = self.rotation_head(x)
        return x, y

    def forward_finetune(self, x):
        x = self.pool(self.encoder(x))
        x = self.flatten(x)
        return self.classifier(x)
