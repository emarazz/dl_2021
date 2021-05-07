# different architecture models

import torch 
from torch import nn
from torch import optim
from torch.nn import functional as F

from helpers import *
from data import *

# most basic architecture inspired from practical 4 adapted to suit this use case with optional maxpool and dropout
class BaseNet(nn.Module):
    def __init__(self, nb_hidden, max_pool=False, dropout_prob=0):
        super().__init__()

        self.conv1 = nn.Conv2d( 2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)
    
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        if max_pool: x = F.max_pool2d(self.conv1(x), kernel_size=3, stride=3)
        x = F.relu(x)
        if max_pool: x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.relu(x)
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
