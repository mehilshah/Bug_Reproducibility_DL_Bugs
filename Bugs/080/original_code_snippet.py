class LR(nn.Module):
    def ___init___(self):
        super(LR, self).___init___()
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(in_features=28*28, out_features=128, bias=True))
    
    def forward(self, x):
        y_p = torch.sigmoid(self.linear(x))
        return y_p

LR_model = LR()
optimizer = torch.optim.SGD(params = LR_model.parameters(), lr=learn_rate)