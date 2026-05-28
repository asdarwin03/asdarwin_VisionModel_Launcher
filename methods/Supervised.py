from torch import nn
import torch.nn.functional as F
from method import Method

class Supervised(Method):
    def __init__(self, encoder, num_classes=10):
        super().__init__(encoder=encoder)
        self.classifier = nn.Linear(self.encoder.dim_out, num_classes)
    
    def forward(self, x, y):
        x = self.encoder(x)
        y_pred = self.classifier(x)
        return F.cross_entropy(y_pred, y)