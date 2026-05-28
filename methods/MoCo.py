#implement MoCo
import torch
from torch import nn
from torch.nn import functional as F
from method import Method

from copy import deepcopy


class MoCo(Method):
    def __init__(self, encoder, queue_size=4096, temperature=0.5, momentum=0.999):
        super().__init__(encoder=encoder)

        self.encoder_q = encoder
        self.encoder_k = deepcopy(encoder)
        self.temperature = temperature
        self.momentum = momentum

        self.projector_q = nn.Sequential( # v2
            nn.Linear(self.encoder_q.dim_out, self.encoder_q.dim_out),
            nn.ReLU(),
            nn.Linear(self.encoder_q.dim_out, self.encoder_q.dim_out),
        )
        self.projector_k = deepcopy(self.projector_q)

        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.projector_k.parameters():
            p.requires_grad = False

        queue = torch.randn(self.encoder_q.dim_out, queue_size)
        self.register_buffer("queue", F.normalize(queue, dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def forward(self, x_q, x_k):
        q = self.projector_q(self.encoder_q(x_q))
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            k = self.projector_k(self.encoder_k(x_k))
            k = F.normalize(k, dim=1)

        l_pos = torch.bmm(q.view(-1, 1, self.encoder_q.dim_out), k.view(-1, self.encoder_q.dim_out, 1)).view(-1, 1)
        l_neg = torch.mm(q, self.queue.clone().detach())

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        self.momentum_update() # update momentum encoder
        self._dequeue_and_enqueue(k)

        return loss

    @torch.no_grad()
    def momentum_update(self):
        m = self.momentum

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue.shape[1]
        self.queue_ptr[0] = ptr