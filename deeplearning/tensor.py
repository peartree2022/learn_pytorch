import torch
#1 2 3
#1 1 1

# 2 3 4
a = torch.tensor([[1, 2],
                 [1, 1]])
b = torch.tensor([[2, 3, 4, 1],
                 [1, 2, 3, 4],
                 [1, 1, 1, 1]])


print(a @ b)
