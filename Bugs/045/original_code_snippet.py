learning_rate = 0.001
num_epochs = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('check 1')
!nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

model = MyModel()

print('check 2')
!nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

model = model.to(device)

print('check 3')
!nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('check 4')
!nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_accuracy = 0.0

    model = model.train()

    print('check 5')
    !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

    ## training step
    for i, (name, output_array, input) in enumerate(trainloader):
        
        output_array = output_array.to(device)
        input = input.to(device)
        comb = torch.zeros(1,1,100,1632).to(device)

        print('check 6')
        !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

        ## forward + backprop + loss
        output = model(input, comb)

        print('check 7')
        !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

        loss = my_loss(output, output_array)

        print('check 8')
        !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

        optimizer.zero_grad()

        print('check 9')
        !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

        loss.backward()

        print('check 10')
        !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

        ## update model params
        optimizer.step()

        print('check 11')
        !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

        train_running_loss += loss.detach().item()

        print('check 12')
        !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

        temp = get_accuracy(output, output_array)

        print('check 13')
        !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

        train_accuracy += temp