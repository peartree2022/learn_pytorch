import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MultiAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads == embed_dim), '嵌入维度必须整除头子数量'

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.concat = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        score = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(score, dim=-1)
        attn = attn @ v

        output = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.concat(output)
        return output

embed_dim = 64
num_heads = 8
batch_size = 32
seq_len = 10

x = torch.rand(batch_size, seq_len, embed_dim)
attention = MultiAttention(embed_dim, num_heads)
output = attention(x)
print(output)
print(output.shape)