import torch
from torch import nn, optim
from torch.autograd import Variable

class VRNNCell(nn.Module):
    def __init__(self):
        super(VRNNCell,self).__init__()
        self.phi_x = nn.Sequential(nn.Embedding(128,64), nn.Linear(64,64), nn.ELU())
        self.encoder = nn.Linear(128,64*2) # output hyperparameters
        self.phi_z = nn.Sequential(nn.Linear(64,64), nn.ELU())
        self.decoder = nn.Linear(128,128) # logits
        self.prior = nn.Linear(64,64*2) # output hyperparameters
        self.rnn = nn.GRUCell(128,64)

    def forward(self, x, hidden):
        x = self.phi_x(x)
        # 1. h => z
        z_prior = self.prior(hidden)
        # 2. x + h => z
        z_infer = self.encoder(torch.cat([x,hidden], dim=1))
        # sampling
        z = Variable(torch.randn(x.size(0),64))*z_infer[:,64:].exp()+z_infer[:,:64]
        z = self.phi_z(z)
        # 3. h + z => x
        x_out = self.decoder(torch.cat([hidden, z], dim=1))
        # 4. x + z => h
        hidden_next = self.rnn(torch.cat([x,z], dim=1),hidden)
        return x_out, hidden_next, z_prior, z_infer

    def calculate_loss(self, x, hidden):
        x_out, hidden_next, z_prior, z_infer = self.forward(x, hidden)
        # 1. logistic regression loss
        loss1 = nn.functional.cross_entropy(x_out, x) 
        # 2. KL Divergence between Multivariate Gaussian
        mu_infer, log_sigma_infer = z_infer[:,:64], z_infer[:,64:]
        mu_prior, log_sigma_prior = z_prior[:,:64], z_prior[:,64:]
        loss2 = (2*(log_sigma_infer-log_sigma_prior)).exp() \
                + ((mu_infer-mu_prior)/log_sigma_prior.exp())**2 \
                - 2*(log_sigma_infer-log_sigma_prior) - 1
        loss2 = 0.5*loss2.sum(dim=1).mean()
        return loss1, loss2, hidden_next
    
    def generate(self, hidden=None, temperature=None):
        if hidden is None:
            hidden=Variable(torch.zeros(1,64))
        if temperature is None:
            temperature = 0.8
        # 1. h => z
        z_prior = self.prior(hidden)
        # sampling
        z = Variable(torch.randn(z_prior.size(0),64))*z_prior[:,64:].exp()+z_prior[:,:64]
        z = self.phi_z(z)
        # 2. h + z => x
        x_out = self.decoder(torch.cat([hidden, z], dim=1))
        # sampling
        x_sample = x = x_out.div(temperature).exp().multinomial(1).squeeze()
        x = self.phi_x(x)
        # 3. x + z => h
        # hidden_next = self.rnn(torch.cat([x,z], dim=1),hidden)
        tc = torch.cat([x,z], dim=1)
        hidden_next = self.rnn(tc,hidden)
        return x_sample, hidden_next
    
    def generate_text(self, hidden=None,temperature=None, n=100):
        res = []
        hidden = None
        for _ in range(n):
            x_sample, hidden = self.generate(hidden,temperature)
            res.append(chr(x_sample.data[0]))
        return "".join(res)
        

# Test
net = VRNNCell()
x = Variable(torch.LongTensor([12,13,14]))
hidden = Variable(torch.rand(3,64))
output, hidden_next, z_infer, z_prior = net(x, hidden)
loss1, loss2, _ = net.calculate_loss(x, hidden)
loss1, loss2

hidden = Variable(torch.zeros(1,64))
net.generate_text()