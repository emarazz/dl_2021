# load data and create Dataset

from dlc_practical_prologue import generate_pair_sets
import torch
from torch.utils.data import DataLoader, Dataset 
from helpers import *

class MNISTPairDataset(Dataset):
    """
    custom Dataset based on the tensor obtained from generate_pair_sets
    """
    def __init__(self, input_data, target, classes):
        self.input_data = input_data
        self.target = target
        self.classes = classes

    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, i):
        input_data = self.input_data[i, :, :, :]
        target = self.target[i]
        classes = self.classes[i, :]
        return input_data, target, classes
    
def get_data(N, batch_size, shuffle):
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N)
    device = get_device()
    # normalization
    mu, std = train_input.mean(), train_input.std()
    train_input = train_input.sub(mu).div(std)
    test_input = test_input.sub(mu).div(std)
    # move data to device
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    train_classes = train_classes.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    test_classes = test_classes.to(device)
    # dataset loader
    train_loader = DataLoader(MNISTPairDataset(train_input, train_target, train_classes), 
                                batch_size = batch_size, shuffle = shuffle)
    test_loader = DataLoader(MNISTPairDataset(test_input, test_target, test_classes), 
                                batch_size = batch_size, shuffle = shuffle)

    return train_loader, test_loader
