import torch
from torch import nn


class TransformerEncoder(nn.Module):
    def __init__(self, D=16*16*3, num_heads=12, dropout=0.1):  # embed_size : (P^2 dot C, P=16, C=3)
        super().__init__()
        self.ln1 = nn.LayerNorm(D)  # for length-variable inputs
        self.attention = nn.MultiheadAttention(embed_dim=D, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=D, out_features=4*D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=4*D, out_features=D),
        )

    def forward(self, x):
        out = self.ln1(x)
        x = x + self.attention(out, out, out)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class visiontransformer(nn.Module):
    def __init__(self, num_classes=10, c_in=3, num_encoders=12, embed_size=16*16*3, img_size=(224, 224), patch_size=16, num_heads=12):
        super().__init__()
        self.P = patch_size  # patch size(16)
        self.L = num_encoders  # number of encoders
        self.N = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # number of tokens = H*W / PP
        self.C = c_in
        self.D = embed_size
        self.class_token = nn.Parameter(torch.randn(1, 1, self.D), requires_grad=True)

        self.patch_embedding = nn.Linear(self.C*(self.P**2), self.D)
        self.position_embedding = nn.Parameter(torch.randn(1, 1+self.N, self.D), requires_grad=True)

        self.encoders = nn.ModuleList([
            TransformerEncoder(D=self.D, num_heads=num_heads) for _ in range(self.L)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, num_classes)
        )


    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        patches = x.unfold(2, self.P, self.P).unfold(3, self.P, self.P)  # unfold(dim, size, stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, C, H/P, W/P, P, P) -> (B, H/P, W/P, C, P, P)
        patches = patches.flatten(start_dim=3)  # -> (B, H/P, W/P, CPP)
        patches = patches.flatten(1, 2)  # -> (B, N=HW/PP, CPP)
        x = self.patch_embedding(patches)
        tokens = self.class_token.repeat(batch_size, 1, 1)  # x가 batch_size개 들어오므로 연산을 위해 각 이미지에 대응하는 class_token을 추가함
        # tokens : (batch_size, 1, D=embed_size)
        # x : (batch_size, N, D)
        x = torch.cat([tokens, x], dim=1)  # (batch_size, 1+N, D)
        x = x + self.position_embedding

        for encoder in self.encoders:
            x = encoder(x)
        x = x[:, 0, :]  # 각 batch에서 class token을 추출해서 임베딩 벡터로 활용
        x = self.mlp_head(x)
        return x

