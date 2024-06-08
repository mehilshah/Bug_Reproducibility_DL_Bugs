import torch
import numpy as np

X_before = np.random.rand(100, 30)
X_tensor = torch.from_numpy(X_before, dtype = float)