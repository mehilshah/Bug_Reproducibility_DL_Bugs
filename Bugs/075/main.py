import torch
import torch.optim as optim
import torch.distributions as D
import numpy as np

num_layers = 8
weights = torch.ones(8,requires_grad=True)
means_coef = torch.tensor(10.,requires_grad=True)
means = torch.tensor(torch.dstack([torch.linspace(1,means_coef.detach().item(),8)]*2).squeeze(),requires_grad=True)
stdevs = torch.tensor(np.abs(np.random.randn(8,2)),requires_grad=True)
parameters = [means_coef]
optimizer1 = optim.SGD(parameters, lr=0.001, momentum=0.9)

num_iter = 101
for i in range(num_iter):
    means = torch.tensor(torch.dstack([torch.linspace(1,means_coef.detach().item(),8)]*2).squeeze(),requires_grad=True)

    mix = D.Categorical(weights)
    comp = D.Independent(D.Normal(means,stdevs), 1)
    gmm = D.MixtureSameFamily(mix, comp)

    optimizer1.zero_grad()
    x = torch.randn(5000,2)#this can be an arbitrary x samples
    loss2 = -gmm.log_prob(x).mean()#-densityflow.log_prob(inputs=x).mean()
    loss2.backward()
    optimizer1.step()

    print(i, means_coef)
    print(means_coef)