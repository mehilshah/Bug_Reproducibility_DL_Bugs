import torch
from torch import nn

kernel_size = 7
stride = 1

# approach 1
data = torch.rand(4, 64, 174, 120)
data1 = data.unfold(3, kernel_size * 2 + 1, stride)
print(data1.shape)

# approach 2
data = torch.rand(4, 64, 174, 120)
unfold = nn.Unfold(3, kernel_size * 2 + 1, stride)
data2 = unfold(data)
print(data2.shape)