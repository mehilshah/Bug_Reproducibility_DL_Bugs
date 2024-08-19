import torch
import torchvision.models as models
import torch.nn as nn
backbone = models.__dict__['densenet169'](pretrained=True)


weight1 = backbone.features.conv0.weight.data.clone()
new_first_layer  = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
with torch.no_grad():
    new_first_layer.weight[:,:3] = weight1

backbone.features.conv0 = new_first_layer
optimizer = torch.optim.SGD(backbone.parameters(), 0.001,
                                 weight_decay=0.1)