import torch 
print(torch.cuda.is_available()) 
torch.randn(1).to("cuda")
