import torch
from torch import nn

class Mymodel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, 10)
        )
    def forward(self, x):
        return self.network(x)
