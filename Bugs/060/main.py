import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random

train_data = torch.utils.data.TensorDataset(torch.randint(0, 200, (1000, 80)), torch.randint(1, 3, (1000,)))
author_dataloader = torch.utils.data.DataLoader(train_data, batch_size=random.choice([32, 64, 128]), shuffle=True)

class Model(nn.Module):
    def __init__(self, tokenize_vocab_count):
        super().__init__()
        self.embd = nn.Embedding(tokenize_vocab_count+1, 300)
        self.embd_dropout = nn.Dropout(0.3)
        self.LSTM = nn.LSTM(input_size=300, hidden_size=100, dropout=0.3, batch_first=True)
        self.lin1 = nn.Linear(100, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin_dropout = nn.Dropout(0.8)
        self.lin3 = nn.Linear(512, 3)

    def forward(self, inp):
        inp = self.embd_dropout(self.embd(inp))
        inp, (h_t, h_o) = self.LSTM(inp)
        h_t = F.relu(self.lin_dropout(self.lin1(h_t)))
        h_t = F.relu(self.lin_dropout(self.lin2(h_t)))
        out = F.softmax(self.lin3(h_t))
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(200).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

EPOCH = 10

for epoch in range(0, EPOCH):
    for feature, target in tqdm(author_dataloader):
        train_loss = loss_fn(model(feature.to(device)).view(-1,  3), target.to(device))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    print(f"epoch: {epoch + 1}\tTrain Loss : {train_loss}")