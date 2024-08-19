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
