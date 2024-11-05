import torch
import torch.nn as nn
from torchvisision import models


class ResNet18(nn.Module):

    def __init__(self, num_classes: int, hidden_dim: int, dropout: int= 0.2, unfrozen_layers: int= 4):
        super(ResNet18, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.droopout = dropout
        self.unfrozen_layers = -unfrozen_layers

        self.base_model = models.resnet18(pretrained=True)
        
        for params in self.base_model.parameters():
            params.requires_grad = False
            
        for layer in list(self.base_model.children())[self.unfrozen_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                        
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, num_classes)
        )
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x