import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def Iris_Reader(dataset):
    train_data, test_data, train_label, test_label = train_test_split(dataset.data, dataset.target, test_size=0.4)

    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.transform(test_data)
    
    return torch.FloatTensor(train_data), torch.LongTensor(train_label), torch.FloatTensor(test_data), torch.LongTensor(test_label)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        #4*3*3 network
        self.model = nn.Sequential(
            nn.Linear(4,3),
            nn.ReLU(),

            nn.Linear(3,3),
        )
        
        #SGD
        self.optimiser = torch.optim.SGD(self.parameters(), lr = 0.1)
        
        #MSE LOSS_FUNCTION
        self.loss_fn = nn.CrossEntropyLoss()

        self.counter = 0
        self.progress = []

    def forward(self, input):
        return self.model(input)
    
    def train(self, input, target):
        output = self.forward(input)

        loss = self.loss_fn(output, target)

        self.counter += 1
        self.progress.append(loss.item())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
    # plot loss
    def plot_loss(self):
        plt.figure(dpi=100)
        plt.ylim([0,1.0])
        plt.yticks([0, 0.25, 0.5, 1.0])
        plt.scatter(x = [i for i in range(len(self.progress))], y = self.progress, marker = '.', alpha = 0.2)
        plt.show()

C = Classifier()
epochs = 10
dataset = datasets.load_iris()

for epoch in range(epochs):
    train_data, train_label, _, _ = Iris_Reader(dataset)
    for i, j in zip(train_data, train_label):
        C.train(i, j)

score = 0
num = 0
# for epoch in range(epochs):
_, _, test_data, test_label = Iris_Reader(dataset)
for i,j in zip(test_data, test_label):
    output = C.forward(i).detach().argmax()
    if output == j:
        # print(C.forward(i).detach(), j)
        score += 1
    num += 1
print(score, num, round(score/num, 3))