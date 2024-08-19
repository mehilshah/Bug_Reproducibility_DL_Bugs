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