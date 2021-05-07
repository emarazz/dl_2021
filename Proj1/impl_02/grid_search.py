import torch
from torch import nn
from torch.nn import functional as F 

from data import *
from helpers import *
from models import *
from train import *

BINARY_SEARCH_ITERATIONS = 3
NUMBER_OF_ROUNDS = 5

def compute_center(two_elements_list):
    return (two_elements_list[0]+two_elements_list[1])/2

def binary_step(two_elements_list, left):
    if left:
        return [two_elements_list[0], compute_center(two_elements_list)]
    else:
        return [compute_center(two_elements_list), two_elements_list[1])]

def binary_search_BaseNet(hidden_layers, batch_sizes, epochs, etas, dropout_probabilities, max_pooling=False):
    device = get_device()
    lowest_error_rate = 0

    for bsi in range(BINARY_SEARCH_ITERATIONS):
        assert(len(hidden_layers) == 2)
        assert(len(batch_sizes) == 2)
        assert(len(epochs) == 2)
        assert(len(etas) == 2)
        assert(len(dropout_probabilities) == 2)

        hidden_layers = [int(hl) for hl in hidden_layers]
        batch_sizes = [int(bs) for bs in batch_sizes]
        epochs = [int(e) for e in epochs]

        for hl in hidden_layers:
            for bs in batch_sizes:
                for e in epochs:
                    for eta in etas:
                        error_rate = 0
                        for r in range(NUMBER_OF_ROUNDS):
                            model = BaseNet(hl, max_pooling)
                            train_loader, test_loader = get_data(N=1000, batch_size=2**bs, shuffle=True)
                            _, er, _ = train_BaseNet(model, e, eta, train_loader, test_loader)
                            error_rate += er[-1]
                            del model
                        averaged_error_rate = error_rate/NUMBER_OF_ROUNDS
                        if averaged_error_rate < lowest_error_rate:
                            used_hl = hl
                            used_bs = bs
                            used_e = e
                            used_eta = eta

        if used_hl == hidden_layers[0]:
            hidden_layers = binary_step(hidden_layers, True)
        else:
            hidden_layers = binary_step(hidden_layers, False)

        if used_bs == batch_sizes[0]:
            batch_sizes = binary_step(batch_sizes, True)
        else:
            batch_sizes = binary_step(batch_sizes, False)

        if used_e == epochs[0]:
            epochs = binary_step(epochs, True)
        else:
            epochs = binary_step(epochs, False)

        if used_eta == etas[0]:
            etas = binary_step(etas, True)
        else:
            etas = binary_step(etas, False)

    for bsi in range(BINARY_SEARCH_ITERATIONS):
        for do in dropout_probabilities:
            error_rate = 0
            for r in range(NUMBER_OF_ROUNDS):
                model = BaseNet(used_hl, max_pooling, do)
                train_loader, test_loader = get_data(N=1000, batch_size=2**used_bs, shuffle=True)
                _, er, _ = train_BaseNet(model, used_e, used_eta, train_loader, test_loader)
                error_rate += er
                del model
            averaged_error_rate = error_rate/NUMBER_OF_ROUNDS
            if averaged_error_rate < lowest_error_rate:
                used_do = do

        if used_do == dropout_probabilities[0]:
            dropout_probabilities = binary_step(dropout_probabilities, True)
        else:
            dropout_probabilities = binary_step(dropout_probabilities, False)
    
    return used_hl, used_bs, used_e, used_eta, used_do 
