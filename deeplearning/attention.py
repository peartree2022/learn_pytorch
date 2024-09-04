import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        #计算q, k, v
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        #计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #计算权重
        x = attn @ v
        return x

attn = Attention(2)
x = torch.rand(1, 4, 2)
output = attn(x)
print(output)
print(output.shape)