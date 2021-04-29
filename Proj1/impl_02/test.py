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
    - try model without dropout and maxpool
    - plot losses
"""

def test_model(model_name, optimizer, n_runs,
                    eta_vals, batch_size_vals, epochs_vals, drop_prob_vals):
    """
    model_name: name of the network - string {"BaseNet", ...}
    optimizer: type of the optimizer - string {"SGD", ...}
    """
    # do grid search to get the "optimal" values

    best_param = grid_search_BaseNet(eta_vals = eta_vals, batch_size_vals = batch_size_vals, epochs_vals = epochs_vals, drop_prob_vals =  drop_prob_vals,
                                        optimizer = optimizer, n_runs = n_runs)
    eta = best_param['eta']
    batch_size = best_param['batch_size']
    epochs = best_param['epochs']
    dropout_prob = best_param['dropout_prob']  

    device = get_device()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    acc_train = [] # train accuracy
    acc_test = [] # test accuracy


    for i in range(0, n_runs):
        print("Run {}".format(i))
        # load data
        train_loader, test_loader = get_data(N = 1000, batch_size = batch_size, shuffle = True )
        
        # select the model
        if model_name == "BaseNet":
            model = BaseNet(nb_hidden = 256, dropout_prob = dropout_prob)
        # add condition for the other models
        model = model.to(device)

        # train & test
        if model_name == "BaseNet":
            # train the model
            losses = []
            _, losses = train_BaseNet(model, train_loader ,criterion, eta, epochs, optimizer)
            # compute train accuracy
            acc_tr = compute_acc_BaseNet(model, train_loader)
            # compute test accuracy
            acc_ts = compute_acc_BaseNet(model, test_loader)
        # add condition for the other models

        acc_train.append(acc_tr)
        acc_test.append(acc_ts)
        print('run n{} -> train accuracy: {:.2f} test accuracy: {:.2f}'.format(n_runs, acc_tr, acc_ts))

        del model
    return acc_train, acc_test, losses 


# "main"

 
n_runs = 3
eta_vals = [1e-3, 1e-2, 1e-1]
drop_prob_vals = [0.2, 0.5, 0.7]
batch_size_vals = [20, 50, 100]
epochs_vals = [10, 30, 50]
acc_train, acc_test, losses = test_model("BaseNet","SGD", n_runs,
                                    eta_vals, batch_size_vals, epochs_vals, drop_prob_vals)
for i in range(0, len(losses)):
    with open("losses.txt", "a") as file:
        text = '{:.4f}\n'.format(losses[i]) 
        file.write(text)