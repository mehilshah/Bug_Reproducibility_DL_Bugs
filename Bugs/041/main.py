import torch

def sigmoid(z):
           return 1 / (1 + torch.exp(-z))
def nueral_net(data,weights,bias):
           return sigmoid( ( data @ weights ) + bias )

def loss_function(prediction,actual,m):
    return (-1/m) * (torch.sum(actual * torch.log(prediction) + (1-actual) 
           * torch.log(1- prediction)))

input_tensor = torch.randn(400,2)
target_tensor = torch.randint(0,2,(400,1))
w = torch.randn(input_tensor.shape[1],1)
b = torch.randn(1,1)

predictions = nueral_net(input_tensor.float() , w, b) #Applying model
loss = loss_function(predictions,target_tensor.unsqueeze(1),400)
dw = (1/400) * torch.dot(input_tensor,(predictions - target_tensor).T)