# different architecture models

import torch 
from torch import nn
from torch import optim
from torch.nn import functional as F

from helpers import *
from data import *

# most basic architecture inspired from practical 4 adapted to suit this use case with optional maxpool and dropout
