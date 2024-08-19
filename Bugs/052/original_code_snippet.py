class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Sequential( 
            nn.Linear(input_size, 8*input_size),
            nn.PReLU() #parametric relu - same as leaky relu except the slope is learned
        )
        self.fc2 = nn.Sequential( 
            nn.Linear(8*input_size, 80*input_size),
            nn.PReLU()
        )
        self.fc3 = nn.Sequential( 
            nn.Linear(80*input_size, 32*input_size),
            nn.PReLU()
        )
        self.fc4 = nn.Sequential( 
            nn.Linear(32*input_size, 4*input_size),
            nn.PReLU()
        )                   
        self.fc = nn.Sequential( 
            nn.Linear(4*input_size, output_size),
            nn.PReLU()
        )
                        

    def forward(self, x, dropout=dropout, batchnorm=batchnorm):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc(x)

        return x
model = Model(input_size, output_size)

if (loss == 'MSE'):
    criterion = nn.MSELoss()
if (loss == 'BCELoss'):
    criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = lr)

model.train()
for epoch in range(num_epochs):
    # Forward pass and loss
    train_predictions = model(train_features)
    print(train_predictions)
    print(train_targets)


    loss = criterion(train_predictions, train_targets)
    
    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()


    train_size = len(train_features)
    train_loss = criterion(train_predictions, train_targets).item() 
    pred = train_predictions.max(1, keepdim=True)[1] 
    correct = pred.eq(train_targets.view_as(pred)).sum().item()
    #train_loss /= train_size
    accuracy = correct / train_size
    print('\nTrain set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, train_size,
        100. * accuracy))