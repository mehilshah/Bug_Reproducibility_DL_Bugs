import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()

    self.conv1 = nn.Conv1d( 1,12, kernel_size=1,stride=5,padding=0)
    self.conv1_drop = nn.Dropout2d()
    self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

    self.fc21 = nn.Linear(198, 1)
    self.fc22 = nn.Linear(198, 1)

    self.fc3 = nn.Linear(1, 198)
    self.fc4 = nn.Linear(198, 1998)

  def encode(self, x):
    h1 = self.conv1(x)
    h1 = self.conv1_drop(h1)
    h1 = self.pool1(h1)
    h1 = F.relu(h1)
    h1 = h1.view(1, -1) # 1 is the batch size
    return self.fc21(h1), self.fc22(h1)
  
  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.rand_like(std)
    return mu + eps*std 
  
  def decode(self, z):
    h3 = F.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h3))
  
  def forward(self, x):
    mu, logvar = self.encode(x.view(-1, 1998))
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

def train(epoch):
      model.train()
      train_loss = 0
      for batch_idx, (data, _) in enumerate(train_loader):
        data = data[None, :, :]
        print(data.size())    # something seems to change between here

        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data) # and here???

        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

for epoch in range(1, 4):
        train(epoch)