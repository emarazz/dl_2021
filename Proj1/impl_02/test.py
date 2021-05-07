import torch
from torch import nn
from torch.nn import functional as F  
from helpers import *
from data import *
from models import *
from train import *
from grid_search import *

NUMBER_OF_EVALUATION_RUNS = 15

def eval_BaseNet(max_pooling=False):
    hidden_layers = [10, 2000]
    batch_sizes = [3, 7]
    epochs = [10, 50]
    etas = [1e-4, 1e-1]
    dropout_probabilities = [0, 0.9]

    hl, bs, e, eta, do = binary_search_BaseNet(hidden_layers, batch_sizes, epochs, etas, dropout_probabilities)

    device = get_device()

    averaged_losses = 0
    averaged_train_error_rate = 0
    averaged_test_error_rate = 0

    filename = "BaseNet"
    if max_pooling:
        filename += "_max_pooling"
    filename += ".txt"

    f = open(filename, "r")
    f.write("{} {}\n".format(e, NUMBER_OF_EVALUATION_RUNS))

    for i in range(NUMBER_OF_EVALUATION_RUNS):
        model = BaseNet(hl, max_pooling, do)
        train_loader, test_loader = get_data(N=1000, batch_size=2**bs, shuffle=True)
        losses, train_error_rates, test_error_rates = train_BaseNet(model, e, eta, train_loader, test_loader)

        averaged_losses += losses[-1]
        averaged_train_error_rate += train_error_rates[-1]
        averaged_test_error_rate += test_error_rates[-1]

        del model

    averaged_losses /= NUMBER_OF_EVALUATION_RUNS
    averaged_train_error_rate /= NUMBER_OF_EVALUATION_RUNS
    averaged_error_rate /= NUMBER_OF_EVALUATION_RUNS

    f.write("hl: {}, bs: 2**{}, e: {}, eta: {}, do: {}, mp: {}\n".format(hl, bs, e, eta, do, max_pooling))
    f.write("loss: {}, train error rate: {}, test error rate: {}".format(averaged_losses, averaged_train_error_rate, averaged_test_error_rate))
    f.close()

eval_BaseNet(False)
eval_BaseNet(True)
