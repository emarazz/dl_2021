import torch

def get_device():
    """
    get the device in which tensors, modules and criterions will be stored
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device