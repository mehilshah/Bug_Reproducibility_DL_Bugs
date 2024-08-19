from multiprocessing import freeze_support
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to('cpu')
opt = torch.optim.Adam(net.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64)

for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        xs, ys = data
        opt.zero_grad()
        preds = net(xs)
        loss = loss(preds,ys)
        loss.backward()
        opt.step()
            # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:  # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        print('epoch {}, loss {}'.format(epoch, loss.item()))
