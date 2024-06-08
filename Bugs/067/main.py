import torch
import torchvision
import torch.nn as nn
import random

lr = random.uniform(0.0001, 0.001)
weight_decay = random.uniform(0.00001, 0.0001)

encoded_dim = 32
encoder = torch.nn.Sequential(
                      torch.nn.Flatten(),
                      torch.nn.Linear(28*28, 256),
                      torch.nn.Sigmoid(),
                      torch.nn.Linear(256, 64),
                      torch.nn.Sigmoid(),
                      torch.nn.Linear(64, encoded_dim)
)
decoder = torch.nn.Sequential(
                      torch.nn.Linear(encoded_dim, 64),
                      torch.nn.Sigmoid(),
                      torch.nn.Linear(64, 256),
                      torch.nn.Sigmoid(),
                      torch.nn.Linear(256, 28*28),
                      torch.nn.Sigmoid(),
                      torch.nn.Unflatten(1, (28,28))
)
autoencoder = torch.nn.Sequential(encoder, decoder)

# Autoencoder 2
# Use torch.nn.Module to create models
# class AutoEncoder(torch.nn.Module):
#     def __init__(self, features: int, hidden: int):
#         # Necessary in order to log C++ API usage and other internals
#         super().__init__()
#         self.encoder = torch.nn.Linear(features, hidden)
#         self.decoder = torch.nn.Linear(hidden, features)

#     def forward(self, X):
#         return self.decoder(self.encoder(X))

#     def encode(self, X):
#         return self.encoder(X)
train_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
                    batch_size=64, shuffle=True)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
for epoch in range(10):
    for idx, (x, _) in enumerate(train_loader):
      x = x.squeeze()
      x = x / x.max()
      x_pred = autoencoder(x) # forward pass
      loss = loss_fn(x_pred, x)
      if idx % 1024 == 0:
        print(epoch, loss.item())
      optimizer.zero_grad()
      loss.backward()         # backward pass
      optimizer.step()
