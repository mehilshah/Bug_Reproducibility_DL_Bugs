import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MixedNetwork(nn.Module):
    def __init__(self):
        super(MixedNetwork, self).__init__()
        
        image_modules = list(models.resnet50().children())[:-1]
        self.image_features = nn.Sequential(*image_modules)

        self.landmark_features = nn.Sequential(
            nn.Linear(in_features=96, out_features=192,bias=False), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.25),
            nn.Linear(in_features=192,out_features=1000,bias=False), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.25))
        
        self.combined_features = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32,1))
        
    def forward(self, image, landmarks):
        a = self.image_features(image)
        print(a.shape)
        b = self.landmark_features(landmarks)
        x = torch.cat((a.view(a.size(0), -1), b.view(b.size(0), -1)), dim=1)
        x = self.combined_features(x)
        x = F.sigmoid(x)
        return x
    
# Input: Images (torch.Size([1, 3, 224, 224])) and landmark features (torch.Size([1, 96]))
image = torch.randn(1, 3, 224, 224)
landmarks = torch.randn(1, 96)

# Instantiate the model
model = MixedNetwork()

# Call the model
output = model(image, landmarks)
