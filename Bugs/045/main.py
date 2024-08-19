import torch

learning_rate = 0.001
num_epochs = 50

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, input, comb):
        out = self.fc(input)
        return out

# Dummy training data loader
train_data = torch.randn(1000, 100)
train_labels = torch.randint(0, 10, (1000,))
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

def my_loss(output, target):
    # Dummy loss function (mean squared error)
    loss = torch.mean((output - target)**2)
    return loss

def get_accuracy(output, target):
    # Dummy accuracy function (classification accuracy)
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('check 1')

model = MyModel()

print('check 2')

model = model.to(device)

print('check 3')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('check 4')

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_accuracy = 0.0

    model = model.train()

    print('check 5')

    ## training step
    for i, (name, output_array, input) in enumerate(trainloader):
        
        output_array = output_array.to(device)
        input = input.to(device)
        comb = torch.zeros(1, 1, 100, 1632).to(device)

        print('check 6')

        ## forward + backprop + loss
        output = model(input, comb)

        print('check 7')

        loss = my_loss(output, output_array)

        print('check 8')

        optimizer.zero_grad()

        print('check 9')

        loss.backward()

        print('check 10')

        ## update model params
        optimizer.step()

        print('check 11')

        train_running_loss += loss.detach().item()

        print('check 12')

        temp = get_accuracy(output, output_array)

        print('check 13')

        train_accuracy += temp
