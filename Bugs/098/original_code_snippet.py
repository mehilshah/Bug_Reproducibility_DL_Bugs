import numpy as np
import torch

X = np.array([[1, 3, 2, 3], [2, 3, 5, 6], [1, 2, 3, 4]])
X = torch.DoubleTensor(X).cuda()

X_split = np.array_split(X.numpy(), 
                         indices_or_sections = 2, 
                         axis = 0)
X_split