import torch
a1, a2 = torch.tensor([1,2], dtype = torch.float64)
b = torch.rand(2, requires_grad = True)

a1 += b.sum()