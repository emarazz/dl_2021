import torch
from torch import nn
from torch.nn import functional as F  
from helpers import *
from data import *
from models import *
from train import *
from grid_search import *

"""
do to: 
    - create a main function 
    - move the parameters in test_model in input (epochs, eta, ...) 
"""

def test_model(model_name, optimizer, epochs, n_runs):
    """
    model_name: name of the network - string {"BaseNet", ...}
    optimizer: type of the optimizer - string {"SGD", ...}
    """
    # do grid search to get the "optimal" values
    """
    eta_vals = [1e-3, 1e-2, 0.5e-2, 0.25e-2, 1e-1, 0.5e-1, 0.25e-1]
    nb_hidd_vals = [16, 32, 64, 128]
    drop_prob_vals = [0.2, 0.5, 0.7]
    best_param = grid_search_BaseNet(eta_vals = eta_vals, nb_hidd_vals = nb_hidd_vals, drop_prob_vals =  drop_prob_vals,
                                        epochs = epochs, optimizer = optimizer, n_runs = n_runs)
    """
    eta = 0.0025 # best_param['eta']
    nb_hidden = 128 # best_param['nb_hidden']
    dropout_prob = 0.2 # best_param['dropout_prob']  

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
            model = BaseNet(nb_hidden = nb_hidden, dropout_prob = dropout_prob)
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

acc_runs, acc_train, acc_test = test_model("BaseNet","SGD", epochs, n_runs)
for 
print(acc_runs / n_runs, acc_train, acc_test)    