import torch
from torch import tensor

a = tensor([[ 1., -5.],
        [ 2., -4.],
        [ 3.,  2.],
        [ 4.,  1.],
        [ 5.,  2.]])

b = tensor([[-1.,  1.],
        [ 1., -1.]], requires_grad=True)

apply_i = lambda x: torch.matmul(x, b)
final = torch.tensor([apply_i(a) for a in a])