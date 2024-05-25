import torch
import timm
from torch import nn
from torchvision import models

#resnet18
class resnet18(nn.Module):
    def __init__(self, num_classes = 2):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def train(self, mode=True):
        super(resnet18, self).train(mode)
        if mode:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
            
    
#vision transformer
class vit_base(nn.Module):
    def __init__(self, num_classes = 2):
        super(vit_base, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(768, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def train(self, mode=True):
        super(vit_base, self).train(mode)
        if mode:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True
    
    
