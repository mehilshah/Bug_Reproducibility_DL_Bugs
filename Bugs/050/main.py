import torch
import torch.nn as nn
import torch.nn.functional as F

# Layer Version 1
# class TransitionLayer(nn.Module):
#   bn = None
#   relu = None
#   conv = None
#   droprate = None

#   def __init__(self, in_size, out_size, drop_rate):
#     super(TransitionLayer, self).__init__()
#     self.bn = nn.BatchNorm2d(in_size)
#     self.relu = nn.ReLU(inplace=True)
#     self.conv = nn.ConvTranspose2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
#     self.droprate = drop_rate

# Layer Version 2
# class TransitionLayer(nn.Module):
#     def __init__(self, in_size, out_size, drop_rate):
#         super(TransitionLayer, self).__init__()
#         self.batch_norm = nn.BatchNorm2d(in_size)
#         self.leaky_relu = nn.LeakyReLU(inplace=True)
#         self.conv2d = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=False)
#         self.dropout = nn.Dropout2d(p=drop_rate)

#     def forward(self, x):
#         out = self.batch_norm(x)
#         out = self.leaky_relu(out)
#         out = self.conv2d(out)
#         out = self.dropout(out)
#         return out

class DenseLayer(nn.Module):
  def __init__(self, in_size, out_size, drop_rate=0.0):
    super(DenseLayer, self).__init__()
    self.bottleneck = nn.Sequential() # define bottleneck layers
    self.bottleneck.add_module('btch1', nn.BatchNorm2d(in_size))
    self.bottleneck.add_module('relu1', nn.ReLU(inplace=True))
    self.bottleneck.add_module('conv1', nn.ConvTranspose2d(in_size, int(out_size/4), kernel_size=1, stride=1, padding=0, bias=False))

    self.basic = nn.Sequential() # define basic block
    self.basic.add_module('btch2', nn.BatchNorm2d(int(out_size/4)))
    self.basic.add_module('relu2', nn.ReLU(inplace=True))
    self.basic.add_module('conv2', nn.ConvTranspose2d(int(out_size/4), out_size, kernel_size=3, stride=1, padding=1, bias=False))

    self.droprate = drop_rate

  def forward(self, input):
    out = self.bottleneck(input)
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    
    out = self.basic(out)
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return torch.cat((x,out), 1)

class DenseBlock(nn.Module):
  def __init__(self, num_layers, in_size, growth_rate, block, droprate=0.0):
    super(DenseBlock, self).__init__()
    self.layer = self._make_layer(block, in_size, growth_rate, num_layers, droprate)

  def _make_layer(self, block, in_size, growth_rate, num_layers, droprate):
    layers = []
    for i in range(num_layers):
      layers.append(block(in_size, in_size-i*growth_rate, droprate))
    return nn.Sequential(*layers)

  def forward(self, input):
    return self.layer(input)

class MGenDenseNet(nn.Module):
  def __init__(self, ngpu, growth_rate=32, block_config=(16,24,12,6), in_size=1024, drop_rate=0.0):
    super(MGenDenseNet, self).__init__()
    self.ngpu = ngpu
    self.features = nn.Sequential()
    self.features.add_module('btch0', nn.BatchNorm2d(in_size))

    block = DenseLayer
    num_features = in_size
    for i, num_layers in enumerate(block_config):
      block = DenseBlock(num_layers=num_layers, in_size=num_features, growth_rate=growth_rate, block=block, droprate=drop_rate) ### Error thrown on this line
      self.features.add_module('denseblock{}'.format(i+1), block)
      num_features -= num_layers*growth_rate

      if i!=len(block_config)-1:
        trans = TransitionLayer(in_size=num_features, out_size=num_features*2, drop_rate=drop_rate)
        self.features.add_module('transitionblock{}'.format(i+1), trans)
        num_features *= 2

    self.features.add_module('convfinal', nn.ConvTranspose2d(num_features, 3, kernel_size=7, stride=2, padding=3, bias=False))
    self.features.add_module('Tanh', nn.Tanh())

  def forward(self, input):
    return self.features(input)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
weights_init = lambda m: torch.nn.init.normal_(m.weight, mean=0, std=0.02)

mGen = MGenDenseNet(1).to(device)
mGen.apply(weights_init)

print(mGen)