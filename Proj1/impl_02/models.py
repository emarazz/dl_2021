# Simple Convolutional Neural Network

import torch 
from torch import nn
from torch.nn import functional as F 
from torch import optim
from helpers import *
from data import *

class BaseNet(nn.Module):
    def __init__(self, nb_hidden, dropout_prob):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)

        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3, stride = 1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 3, stride = 3))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

