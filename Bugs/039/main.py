import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, optimizer, train_loader,  num_epochs, criterion = nn.CrossEntropyLoss()):
  model.train()

  running_loss=0
  correct=0
  total=0

  for epoch in range(num_epochs):
    for i, (x_train, y_train) in enumerate(train_loader):

      x_train = x_train.to(device)
      y_train = y_train.to(device)
        
      y_pred = model(x_train)
      loss = criterion(y_pred, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      
      _, predicted = y_pred.max(1)
      train_loss=running_loss/len(train_loader)


      total += y_train.size(0)
      correct += predicted.eq(y_train).sum().item()
        
    train_loss=running_loss/len(train_loader)
    train_accu=100.*correct/total

    print('Train Loss: %.3f | Train Accuracy: %.3f'%(train_loss,train_accu))

# Model 1
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(64 * 12 * 12, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.max_pool2d(self.dropout1(x), 2)
#         x = torch.flatten(x, 1)
#         x = nn.functional.relu(self.fc1(self.dropout2(x)))
#         x = self.fc2(x)
#         return nn.functional.log_softmax(x, dim=1)

# Model 2
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(28*28, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10)
    
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
  
# Define the data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

# Define the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

# Define the model and optimizer
model = Net()
optimizer = optim.Adam(model.parameters())

train_md = train_model(model, optimizer, train_loader, 10)