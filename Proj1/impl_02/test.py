import torch
from torch import nn
from torch.nn import functional as F  
from helpers import *
from data import *
from models import *
from train import *

"""
do to: 
    - create a main function 
    - move the parameters in test_model in input (epochs, eta, ...) 
"""

def test_model(model_name, optimizer, epochs, eta, n_runs):
    """
    model_name: name of the network - string {"BaseNet", ...}
    optimizer: type of the optimizer - string {"SGD", ...}
    """
    device = get_device()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    acc_runs = []
    acc_train = []
    acc_test = []
    for i in range(0, n_runs):
        print("Run {}".format(i))
        # load data
        train_loader, test_loader = get_data(N = 1000, batch_size = 10 , shuffle = True )
        
        # select the model
        if model_name == "BaseNet":
            model = BaseNet(nb_hidden = 32, dropout_prob = 0.5)
        # add condition for the other models
        model = model.to(device)

        # train & test
        if model_name == "BaseNet":
            # train the model
            acc = train_BaseNet(model, train_loader ,criterion, eta, epochs, optimizer)
            # compute train accuracy
            acc_tr = compute_acc_BaseNet(model, train_loader)
            # compute test accuracy
            acc_ts = compute_acc_BaseNet(model, test_loader)
        # add condition for the other models

        acc_runs.append(acc)
        acc_train.append(acc_tr)
        acc_test.append(acc_ts)
        del model

    return acc_runs, acc_train, acc_test


# "main"

epochs = 100
eta = 1e-2 # learning rate 
n_runs = 3

acc_runs, acc_train, acc_test = test_model("BaseNet","SGD", epochs, eta, n_runs)
print(acc_runs, acc_train, acc_test)