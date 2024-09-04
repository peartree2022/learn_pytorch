import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, max_pos, embed_dim):
        super().__init__()
        PE = torch.zeros(max_pos, embed_dim)
        pos = torch.arange(0, max_pos).unsqueeze(1).float()

        multi_term = torch.arange(0, embed_dim, 2).float()
        multi_term = torch.exp(multi_term * (-math.log(10000) / embed_dim))
        PE[:, 0::2] = torch.sin(pos * multi_term)
        PE[:, 1::2] = torch.cos(pos * multi_term)
        self.register_buffer('PE', PE.unsqueeze(0))

    def forward(self, x):
        return x + self.PE[:, :x.size(1)].clone().detach()

max_pos = 10
embed_dim = 4
posemb = PositionalEncoding(max_pos, embed_dim)

x = torch.zeros(2, 5, embed_dim)
output = posemb(x)
print(output)
print(output.shape)