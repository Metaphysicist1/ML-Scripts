import torch.functional as F
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden1 = nn.Linear(784,128)
        self.hidden2 = nn.Linear(128,64)
        self.output = nn.Linear(64,10)

    def forward(self,x):
        x = 
