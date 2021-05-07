import torch
from torch import nn
from torch.nn import functional as F  

from data import *
from helpers import *

BINARY_SEARCH_ITERATIONS = 3
NUMBER_OF_ROUNDS = 3
NUMBER_OF_EVALUATION_RUNS = 15

class BaseNet(nn.Module):
    def __init__(self, nb_hidden, max_pool=False, dropout_prob=0):
        super().__init__()

        self.max_pool = max_pool

        self.conv1 = nn.Conv2d( 2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.fc1 = nn.Linear(64*6*6, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)
    
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = self.conv1(x)
        if self.max_pool:
            x = F.max_pool2d(x, kernel_size=3, stride=3)
        x = F.relu(x)
        x = self.conv2(x)
        if self.max_pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        x = F.relu(self.fc1(x.view(-1, 64*6*6)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_BaseNet(model, epochs, eta, train_loader, test_loader, eval_mode=False):
    losses = []
    train_error_rates = []
    test_error_rates = []

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(get_device())
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
        
    for _ in range(epochs):
        for input_data, target, _ in iter(train_loader):
            output = model(input_data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss)
        train_error_rates.append(compute_error_rate(model, train_loader))
        test_error_rates.append(compute_error_rate(model, test_loader))

    return losses, train_error_rates, test_error_rates

def compute_center(two_elements_list):
    return (two_elements_list[0]+two_elements_list[1])/2

def binary_step(two_elements_list, left):
    if left:
        return [two_elements_list[0], compute_center(two_elements_list)]
    else:
        return [compute_center(two_elements_list), two_elements_list[1]]

def binary_search_BaseNet(hidden_layers, batch_sizes, epochs, etas, dropout_probabilities, max_pooling=False):
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
                            _, _, er = train_BaseNet(model, e, eta, train_loader, test_loader)
                            error_rate += er[-1]
                            del model
                        averaged_error_rate = error_rate/NUMBER_OF_ROUNDS
                        if averaged_error_rate < lowest_error_rate:
                            used_hl = hl
                            used_bs = bs
                            used_e = e
                            used_eta = eta

                        print("bsi: {}, hl: {}, bs: 2**{}, e: {}, eta: {}, mp: {} -> er: {}\n".format(bsi, hl, bs, e, eta, max_pooling, averaged_error_rate))

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
                _, _, er = train_BaseNet(model, used_e, used_eta, train_loader, test_loader)
                error_rate += er
                del model
            averaged_error_rate = error_rate/NUMBER_OF_ROUNDS
            if averaged_error_rate < lowest_error_rate:
                used_do = do

            print("bsi: {}, hl: {}, bs: 2**{}, e: {}, eta: {}, do: {}, mp: {} -> er: {}\n".format(bsi, used_hl, used_bs, used_e, used_eta, do, max_pooling, averaged_error_rate))

        if used_do == dropout_probabilities[0]:
            dropout_probabilities = binary_step(dropout_probabilities, True)
        else:
            dropout_probabilities = binary_step(dropout_probabilities, False)
    
    return used_hl, used_bs, used_e, used_eta, used_do 

def eval_BaseNet(max_pooling=False):
    hidden_layers = [10, 2000]
    batch_sizes = [3, 7]
    epochs = [10, 40]
    etas = [1e-3, 1e-1]
    dropout_probabilities = [0, 0.9]

    hl, bs, e, eta, do = binary_search_BaseNet(hidden_layers, batch_sizes, epochs, etas, dropout_probabilities)

    filename = "BaseNet"
    if max_pooling:
        filename += "_max_pooling"
    filename += "_parameters.txt"

    f = open(filename, "r")
    f.write("hl: {}, bs: 2**{}, e: {}, eta: {}, do: {}, mp: {}\n".format(hl, bs, e, eta, do, max_pooling))
    f.close()

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

        f.write(" ".join([str(l) for l in losses])+"\n")
        f.write(" ".join([str(er) for er in train_error_rates])+"\n")
        f.write(" ".join([str(er) for er in test_error_rates])+"\n")

        averaged_losses += losses[-1]
        averaged_train_error_rate += train_error_rates[-1]
        averaged_test_error_rate += test_error_rates[-1]

        del model

    averaged_losses /= NUMBER_OF_EVALUATION_RUNS
    averaged_train_error_rate /= NUMBER_OF_EVALUATION_RUNS
    averaged_error_rate /= NUMBER_OF_EVALUATION_RUNS

    f.write("loss: {}, train error rate: {}, test error rate: {}".format(averaged_losses, averaged_train_error_rate, averaged_test_error_rate))
    f.close()
