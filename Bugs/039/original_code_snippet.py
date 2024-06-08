def train_model(model, optimizer, train_loader,  num_epochs, criterion=criterion):
  
  total_epochs = notebook.tqdm(range(num_epochs))

  model.train()

  running_loss=0
  correct=0
  total=0

  for epoch in total_epochs:
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