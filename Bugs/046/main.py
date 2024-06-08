from torch.autograd import Variable
import torch
import numpy as np

train_x = np.asarray([1,2,3,4,5,6,7,8,9,10,5,4,6,8,5,2,1,1,6])
train_y = train_x * 2

X = Variable(torch.from_numpy(train_x).type(torch.FloatTensor), requires_grad = False).view(19, 1)
y = Variable(torch.from_numpy(train_y).type(torch.FloatTensor), requires_grad = False)
from torch import nn


lr = nn.Linear(19, 1) 

loss = nn.MSELoss()
optimizer = torch.optim.SGD(lr.parameters(), lr = 0.01)
output = lr(X) #error occurs here