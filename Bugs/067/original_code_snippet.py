
wEncoder = torch.randn(D,1, requires_grad=True)
wDecoder = torch.randn(1,D, requires_grad=True)
bEncoder = torch.randn(1, requires_grad=True)
bDecoder = torch.randn(1,D, requires_grad=True)

D = 2
x = torch.rand(100,D)
x[:,0] = x[:,0] + x[:,1]
x[:,1] = 0.5*x[:,0] + x[:,1]

loss_fn = nn.MSELoss()
optimizer = optim.SGD([x[:,0],x[:,1]], lr=0.01)
losses = []
for epoch in range(1000):
    running_loss = 0.0
    inputs = x_reconstructed
    targets = x
    loss=loss_fn(inputs,targets)
    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    running_loss += loss.item() 
    epoch_loss = running_loss / len(data)
    losses.append(running_loss)