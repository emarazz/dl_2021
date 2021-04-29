import torch
from torch import nn
from torch.nn import functional as F 
from train import *
from helpers import *
from models import *
from data import *

def grid_search_BaseNet(eta_vals, batch_size_vals, epochs_vals, drop_prob_vals,
                        optimizer, n_runs):
    device = get_device()
    """
    eta_vals : values of the learning rate
    drop_prob_vals : values of the dropout probability
    batch_size_vals : values of batch_size 
    epochs_vals : numbers of epochs
    """
    # initialize the file
    with open("grid_search.txt", "a") as file:
        text = '  n  |  eta  |  batch size  |  nb epochs  |  dropout  || accuracy\n' 
        file.write(text)
    count = 0
    best_acc = 0 
    best_param = {'eta' : 0, 'batch_size' : 0, 'epochs' : 0, 'dropout_prob' : 0}
    nb_hidd = 256
    # TODo:
    # learning rate, batch_size 5.2 7/17, and number of epochs(?) and dropout. Don't focus on the number of layers
    # plot all the losses (once the optimal param have been found)
    # try without dropout maxpool
    # add batch normalization
    # weight 
    for eta in eta_vals:
        for batch_size in batch_size_vals:
            for epochs in epochs_vals:
                for dropout_prob in drop_prob_vals:
                    acc = 0
                    for n in range(0,n_runs):
                        model = BaseNet(nb_hidd, dropout_prob).to(device)
                        criterion = nn.CrossEntropyLoss().to(device)
                        # load data
                        train_loader, test_loader = get_data(N = 1000, batch_size = batch_size, shuffle = True)
                        # train the model
                        train_BaseNet(model, train_loader, criterion, eta, epochs, optimizer)
                        # compute test accuracy
                        acc += compute_acc_BaseNet(model, test_loader)
                        del model
                    acc = acc / n_runs 
                    if best_acc < acc:
                        best_acc = acc
                        best_param['eta'] = eta
                        best_param['batch_size'] = batch_size
                        best_param['epochs'] = epochs
                        best_param['dropout_prob'] = dropout_prob
                    count += 1
                    with open("grid_search.txt", "a") as file:
                        text = ' {} | {} | {} | {} | {} || {:.3f} \n'.format(count, best_param['eta'], best_param['batch_size'], best_param['epochs'], best_param['dropout_prob'], best_acc) 
                        file.write(text)
    return best_param 

