import torch
from torch import nn
from torch.nn import functional as F  

from data import *
from helpers import *
from time import time

BINARY_SEARCH_ITERATIONS = 4
NUMBER_OF_SEARCH_RUNS = 1
NUMBER_OF_EVALUATION_RUNS = 10
"""
optimal parameters: 
    nb_hidden1 = 512
    nb_hidden2 = 512
    do = 0.125
    eta = 0.075
    bs = 6
"""
class BaseNet(nn.Module):
    def __init__(self, nb_hidden1, nb_hidden2, dropout_prob=0):
        super().__init__()
        
        # fully convoluted
        self.conv1 = nn.Conv2d( 2, 32, kernel_size=5) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3) 
        
        # fully connected 
        self.fc1 = nn.Linear(256, nb_hidden1)
        self.fc2 = nn.Linear(nb_hidden1, nb_hidden2)
        self.fc3 = nn.Linear(nb_hidden2, 2)
        
        self.max_pool = nn.MaxPool2d(kernel_size = 2)
        self.dropout = nn.Dropout(dropout_prob)
        # batch norms
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.batchNorm2 = nn.BatchNorm1d(nb_hidden1)
        self.batchNorm3 = nn.BatchNorm1d(nb_hidden2)
        self.batchNorm4 = nn.BatchNorm2d(32)
        self.batchNorm5 = nn.BatchNorm2d(64)

        self.relu = nn.LeakyReLU(0.1)
    def forward(self, x): 
        x = self.relu(self.batchNorm4(self.conv1(x)))           # 32x10x10
        x = self.relu(self.batchNorm5(self.conv2(x)))           # 64x6x6
        x = self.conv3(x)       # 64x4x4
        x = self.batchNorm1(x)  # 64x4x4
        x = self.relu(x)           # 64x4x4
        x = self.max_pool(x)    # 64x2x2
        x = self.fc1(x.view(-1, 256)) # nb_hidden1x1
        x = self.batchNorm2(x)  # nb_hidden1x1
        x = self.relu(x)           # nb_hidden1x1
        x = self.dropout(x)     # nb_hidden1x1
        x = self.fc2(x)         # nb_hidden2x1
        x = self.batchNorm3(x)  # nb_hidden2x1
        x = self.relu(x)           # nb_hidden2x1
        x = self.dropout(x)     # nb_hidden2x1
        x = self.fc3(x)         # 2x1
        return x

def train_BaseNet(model, eta, train_loader, test_loader, eval_mode=False, optim = 'Adam'):
    losses = []
    train_error_rates = []
    test_error_rates = []
    epochs = 30

    device = get_device() 
    model  = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)
        
    for e in range(epochs):
        for input_data, target, _ in iter(train_loader):
            output = model(input_data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss)
        train_error_rates.append(compute_error_rate(model, train_loader))
        test_error_rates.append(compute_error_rate(model, test_loader))
        
        if ((e%10) == 0):
            print(get_str_results(e, train_losses[-1], train_acc[-1], test_acc[-1]))
                
    print(get_str_results(e, train_losses[-1], train_acc[-1], test_acc[-1]))
    print(60*'-')

    return losses, train_error_rates, test_error_rates

def compute_center(two_elements_list):
    return (two_elements_list[0]+two_elements_list[1])/2

def binary_step(two_elements_list, left):
    if left:
        return [two_elements_list[0], compute_center(two_elements_list)]
    else:
        return [compute_center(two_elements_list), two_elements_list[1]]

def binary_search_BaseNet(hidden_layers1, hidden_layers2, batch_sizes, etas, dropout_probabilities):
    lowest_error_rate = float('inf')

    used_hl = -1
    used_h2 = -1
    used_bs = -1
    used_eta = -1
    used_do = -1

    filename = "BaseNet_binarysearch.txt"
    f = open(filename, "w")
    for bsi in range(BINARY_SEARCH_ITERATIONS):
        assert(len(hidden_layers1) == 2)
        assert(len(hidden_layers2) == 2)
        assert(len(batch_sizes) == 2)
        assert(len(dropout_probabilities) == 2)

        hidden_layers1 = [int(hl) for hl in hidden_layers1]
        hidden_layers2 = [int(h2) for h2 in hidden_layers2]
        batch_sizes = [int(bs) for bs in batch_sizes]

        for hl in hidden_layers1:
            for h2 in hidden_layers2:
                for bs in batch_sizes:
                     for eta in etas:
                        for do in dropout_probabilities:
                            last_time = time()
                            error_rate = 0
                            for r in range(NUMBER_OF_SEARCH_RUNS):
                                model = BaseNet(hl, h2, do)
                                train_loader, test_loader = get_data(N=1000, batch_size=2**bs, shuffle=True)
                                _, _, er = train_BaseNet(model, eta, train_loader, test_loader)
                                error_rate += er[-1]
                                del model
                            averaged_error_rate = error_rate/NUMBER_OF_SEARCH_RUNS
                            if averaged_error_rate < lowest_error_rate:
                                lowest_error_rate = averaged_error_rate
                                used_hl = hl
                                used_h2 = h2
                                used_bs = bs
                                used_eta = eta
                                used_do = do

                            print("bsi: {:1.0f}, hl: {:4.0f}, h2: {:4.0f}, bs: 2**{:1.0f}, eta: {:.5f}, do: {:.5f}-> er: {:.3f} in about {:.2f}sec".format(bsi, hl, h2, bs, eta, do, averaged_error_rate, time()-last_time))
                            f.write("bsi: {:1.0f}, hl: {:4.0f}, h2: {:4.0f}, bs: 2**{:1.0f}, eta: {:.5f}, do: {:.5f}-> er: {:.3f} in about {:.2f}sec\n".format(bsi, hl, h2, bs, eta, do, averaged_error_rate, time()-last_time))
        
        if used_hl == hidden_layers1[0]:
            hidden_layers1 = binary_step(hidden_layers1, True)
        else:
            hidden_layers1 = binary_step(hidden_layers1, False)
        
        if used_h2 == hidden_layers2[0]:
            hidden_layers2 = binary_step(hidden_layers2, True)
        else:
            hidden_layers2 = binary_step(hidden_layers2, False)
        
        if used_bs == batch_sizes[0]:
            batch_sizes = binary_step(batch_sizes, True)
        else:
            batch_sizes = binary_step(batch_sizes, False)


        if used_eta == etas[0]:
            etas = binary_step(etas, True)
        else:
            etas = binary_step(etas, False)

        if used_do == dropout_probabilities[0]:
            dropout_probabilities = binary_step(dropout_probabilities, True)
        else:
            dropout_probabilities = binary_step(dropout_probabilities, False)
    f.close()
    return used_hl, used_h2,  used_bs, used_eta, used_do 

def eval_BaseNet():
    hidden_layers1 = [10, 512]
    hidden_layers2 = [10, 512]
    batch_sizes = [4, 7]
    etas = [1e-3, 1e-1]
    dropout_probabilities = [0, 0.5]

    hl, h2, bs, eta, do = binary_search_BaseNet(hidden_layers1, hidden_layers2, batch_sizes, etas, dropout_probabilities)

    filename = "BaseNet_parameters.txt"

    f = open(filename, "w")
    f.write("hl: {}, h2: {}, bs: 2**{}, eta: {}, do: {}\n".format(hl, h2, bs, eta, do))
    f.close()

    filename = "BaseNet_scores.txt"

    f = open(filename, "w")
    f.write("{} {}\n".format(30, NUMBER_OF_EVALUATION_RUNS))

    averaged_losses = 0
    averaged_train_error_rate = 0
    averaged_test_error_rate = 0

    for i in range(NUMBER_OF_EVALUATION_RUNS):
        model = BaseNet(hl, h2, do)
        train_loader, test_loader = get_data(N=1000, batch_size=2**bs, shuffle=True)
        losses, train_error_rates, test_error_rates = train_BaseNet(model, eta, train_loader, test_loader)

        f.write(" ".join([str(l.item()) for l in losses])+"\n")
        f.write(" ".join([str(er) for er in train_error_rates])+"\n")
        f.write(" ".join([str(er) for er in test_error_rates])+"\n")

        averaged_losses += losses[-1]
        averaged_train_error_rate += train_error_rates[-1]
        averaged_test_error_rate += test_error_rates[-1]

        del model

    averaged_losses /= NUMBER_OF_EVALUATION_RUNS
    averaged_train_error_rate /= NUMBER_OF_EVALUATION_RUNS
    averaged_test_error_rate /= NUMBER_OF_EVALUATION_RUNS

    print("loss: {}, train error rate: {}, test error rate: {} saved to file\n".format(averaged_losses, averaged_train_error_rate, averaged_test_error_rate))

    f.write("loss: {}, train error rate: {}, test error rate: {}\n".format(averaged_losses, averaged_train_error_rate, averaged_test_error_rate))
    f.close()

