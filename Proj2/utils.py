import torch
from torch import Tensor

# Set autograd off
torch.set_grad_enabled(False)

def is_inside(input, cols=2):
    """
    Returns 0 if the input is outside a disk centered at (0.5, 0.5) of radius 1 / sqrt(2*pi)) and 1 if inside.
    """

    pi = torch.acos(torch.tensor(-1.)) # Calculate pi :p
    center = torch.tensor([0.5, 0.5])
    radius = 1/(torch.sqrt(2 * pi))

    output = torch.zeros(input.size(0), cols, dtype=torch.long)
    mask = torch.norm(input.view(-1,2) - center, dim=1) < radius
    output[mask] = 1
    one_hot_labels = torch.eye(cols)[output[:,0]] # One hot encoding

    return one_hot_labels