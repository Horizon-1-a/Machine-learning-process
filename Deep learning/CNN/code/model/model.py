import numpy as np
import torch
from torch import nn
from data import get

class CNN (nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size= 3, padding= 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride= 2), 
            nn.Conv2d(32, 64, kernel_size= 3, padding= 1), 
            nn.MaxPool2d(kernel_size= 2, stride = 2), 
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features= 64 * 7 * 7, out_features= hidden_size, bias= True), 
            nn.ReLU(), 
            nn.Linear(in_features= 128, out_features= hidden_size, bias= True)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x;
