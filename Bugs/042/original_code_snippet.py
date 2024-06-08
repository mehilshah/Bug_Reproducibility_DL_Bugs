import torch
from torch import nn, optim
import torch.nn.functional as F
X_train_t = torch.tensor(X_train).float()
X_test_t = torch.tensor(X_test).float()
y_train_t = torch.tensor(y_train).long().reshape(y_train_t.shape[0], 1)
y_test_t = torch.tensor(y_test).long().reshape(y_test_t.shape[0], 1)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(22, 10)
        self.fc2 = nn.Linear(10, 1)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x

model = Classifier()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 2000
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    # training loss
    optimizer.zero_grad()

    log_ps = model(X_train_t)
    loss = criterion(log_ps, y_train_t.type(torch.float32))
    loss.backward()
    optimizer.step()
    train_loss = loss.item()

    # test loss
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        log_ps = model(X_test_t)
        test_loss = criterion(log_ps, y_test_t.to(torch.float32))
        ps = torch.exp(log_ps)

    train_losses.append(train_loss/len(X_train_t))
    test_losses.append(test_loss/len(X_test_t))
    
    if (e % 100 == 0):
        print("Epoch: {}/{}.. ".format(e, epochs),
          "Training Loss: {:.3f}.. ".format(train_loss/len(X_train_t)),
          "Test Loss: {:.3f}.. ".format(test_loss/len(X_test_t)))