import torch
import torch.nn as nn
import random 

# select a random value between 0.01 and 0.001
learn_rate = random.uniform(0.001, 0.01)

class LR(nn.Module):
    def ___init___(self):
        super(LR, self).___init___()
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(in_features=28*28, out_features=128, bias=True))
    
    def forward(self, x):
        y_p = torch.sigmoid(self.linear(x))
        return y_p

LR_model = LR()
optimizer = torch.optim.SGD(params = LR_model.parameters(), lr=learn_rate)