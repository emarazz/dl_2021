import torch
from torch import nn
from torch.nn import functional as F 
from train import *
from helpers import *
from models import *
from data import *

def grid_search_BaseNet(eta_vals, nb_hidd_vals, drop_prob_vals,
                        epochs, optimizer, n_runs):
    device = get_device()
    """
    eta_vals : values of the learning rate
    nb_hidd_vals : number of hidden layers
    drop_prob_vals : values of the dropout probability
    """
    count = 0
    best_acc = 0 
    best_param = {'eta' : 0, 'nb_hidden' : 0, 'dropout_prob' : 0}
    for eta in eta_vals:
        for nb_hidd in nb_hidd_vals:
            for dropout_prob in drop_prob_vals:
                acc = 0
                for n in range(0,n_runs):
                    model = BaseNet(nb_hidd, dropout_prob).to(device)
                    criterion = nn.CrossEntropyLoss().to(device)
                    # load data
                    train_loader, test_loader = get_data(N = 1000, batch_size = 10, shuffle = True)
                    # train the model
                    train_BaseNet(model, train_loader, criterion, eta, epochs, optimizer)
                    # compute test accuracy
                    acc += compute_acc_BaseNet(model, test_loader)
                    del model
                acc = acc / n_runs 
                if best_acc < acc:
                    best_acc = acc
                    best_param['eta'] = eta
                    best_param['nb_hidden'] = nb_hidd
                    best_param['dropout_prob'] = dropout_prob
                count += 1
                with open("grid_search.txt", "a") as file:
                    text = 'comb n {} -> eta: {} nb_hidden: {} droput_prob: {} acc: {:.4f} \n'.format(count, best_param['eta'], best_param['nb_hidden'], best_param['dropout_prob'], best_acc)
                    file.write(text)
    return best_param 

