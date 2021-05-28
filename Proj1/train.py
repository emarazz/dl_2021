import torch
from torch import nn
from torch.nn import functional as F  

from data import *
from helpers import *
from models import *
from time import time
import os

BINARY_SEARCH_ITERATIONS = 4
NUMBER_OF_SEARCH_RUNS = 1
NUMBER_OF_EVALUATION_RUNS = 15


def train_Net(model, eta, epochs, train_loader, test_loader, optim = 'Adam', alpha = 1, beta = 1, print_results=True):
    model.train()

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    device = get_device() 
    model  = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=eta, betas=(0.99, 0.999))
        
    for e in range(epochs):
        model.train()
        if model.get_aux_info() == False:
            for train_input, train_target, _ in iter(train_loader):
                output = model(train_input)
                loss = criterion(output, train_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif model.get_aux_info() == True:
            for input_data, target, class_data in iter(train_loader):
                output_aux1, output_aux2, output = model(input_data)
                # auxiliary losses
                loss_aux1 = criterion(output_aux1, class_data[:,0])
                loss_aux2 = criterion(output_aux2, class_data[:,1])
                # total losses
                loss = alpha * criterion(output, target) + beta * ( loss_aux1 + loss_aux2 )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            print('[error] no information on auxiliary losses')
            return
        
        train_losses.append(eval_model(model=model, data_loader=train_loader, alpha=alpha, beta=beta))
        test_losses.append(eval_model(model=model, data_loader=test_loader, alpha=alpha, beta=beta))
        train_accs.append(compute_acc(model=model,data_loader=train_loader)) 
        test_accs.append(compute_acc(model=model,data_loader=test_loader)) 
        
        if print_results:
            if (((e+1)%10) == 0) or e == 0:
                print(get_str_results(epoch=e+1, train_loss= train_losses[-1], test_loss=test_losses[-1] , train_acc= train_accs[-1], test_acc=test_accs[-1]))
    
    if print_results:           
        print(70*'-')

    return train_losses, test_losses, train_accs, test_accs
    
def compute_center(two_elements_list):
    return (two_elements_list[0]+two_elements_list[1])/2

def binary_step(two_elements_list, left):
    if left:
        return [two_elements_list[0], compute_center(two_elements_list)]
    else:
        return [compute_center(two_elements_list), two_elements_list[1]]

def binary_search_Net(cls, nb_hidden1 = [64, 512], nb_hidden2 = [64, 512], nb_hidden3 = [0], 
                            dropout_probabilities = [0], log2_batch_sizes = [6], etas = 0.01, epochs = 30,
                            optim = 'Adam', alpha = 1, beta = 1):
    """
    binary search for the following hyperparameters 
    outputs the combination of hyperparameters with higher accuracy
    
    hyperparameters
        cls:                    class of the model
        nb_hidden:              range of number of hidden units
        droput_probabilities:   range of dropout probabilities
        log2_batch_size:        range of batch size (base 2 exponent) 
        etas:                   range of learning rates
    parameters
        epochs:                 number of epochs
        optim:                  optimizer in {'SGD','Adam'}
        alpha, beta:            weight of the losses in models with auxiliary losses
    """
    
    highest_acc = float('-inf')

    used_h1 = -1
    used_h2 = -1
    used_h3 = -1
    used_log2_bs = -1
    used_eta = -1
    used_do = -1

    filename = cls.__name__ + '_binarysearch.txt'
    if os.path.exists(filename):
        os.remove(filename)

    for bsi in range(BINARY_SEARCH_ITERATIONS):
        
        len_h1 = len(nb_hidden1) == 2
        len_h2 = len(nb_hidden2) == 2
        len_h3 = len(nb_hidden3) == 2
        len_log2_bs = len(log2_batch_sizes) == 2
        len_etas = len(etas) == 2
        len_do = len(dropout_probabilities) == 2


        nb_hidden1 = [int(h1) for h1 in nb_hidden1]
        nb_hidden2 = [int(h2) for h2 in nb_hidden2]  
        nb_hidden3 = [int(h3) for h3 in nb_hidden3] 
        log2_batch_sizes = [int(log2_bs) for log2_bs in log2_batch_sizes]

        for h1 in nb_hidden1:
            for h2 in nb_hidden2:
                for h3 in nb_hidden3:
                    for log2_bs in log2_batch_sizes:
                        for eta in etas:
                            for do in dropout_probabilities:
                                last_time = time()
                                acc_cum = 0
                                for r in range(NUMBER_OF_SEARCH_RUNS):
                                    model = cls(h1, h2, h3, do) 
                                    train_loader, test_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True, validation = True, val_size = 800)
                                    _, _, _, test_accs = train_Net(model=model, eta=eta, epochs=epochs, train_loader=train_loader, test_loader=test_loader,
                                                                    optim = optim, alpha = alpha, beta = beta)
                                    acc_cum += test_accs[-1]
                                    del model
                                averaged_acc = acc_cum/NUMBER_OF_SEARCH_RUNS
                                print(averaged_acc , highest_acc)   
                                if averaged_acc > highest_acc:
                                    highest_acc = averaged_acc
                                    used_h1 = h1
                                    used_h2 = h2
                                    used_h3 = h3
                                    used_log2_bs = log2_bs
                                    used_eta = eta
                                    used_do = do
                                
                                # print('-'*70)
                                print("bsi: {:2d}, h1: {}, h2: {}, h3: {}, do: {:.3f}, bs: 2**{}, eta: {:.4E} -> er: {:.4f} in about {:.2f}sec".format(bsi, h1, h2, h3, do, log2_bs, eta, averaged_acc, time()-last_time))
                                print('='*70)
                                # print('='*70 +'\n' + '='*70)
                                with open(filename, "a") as f:
                                    f.write("bsi: {:2d}, h1: {}, h2: {},  h3: {}, do: {:.3f}, bs: 2**{}, eta: {:.4E} -> er: {:.4f} in about {:.2f}sec\n".format(bsi, h1, h2, h3, do, log2_bs, eta, averaged_acc, time()-last_time))
        
        if not (len_h1 or len_h2 or len_log2_bs or len_etas or len_do): # if binary search is not possible -> break the loop
            break

        if len_h1:
            if used_h1 == nb_hidden1[0]:
                nb_hidden1 = binary_step(nb_hidden1, True)
            else:
                nb_hidden1 = binary_step(nb_hidden1, False)
        
        if  len_h2:
            if used_h2 == nb_hidden2[0]:
                nb_hidden2 = binary_step(nb_hidden2, True)
            else:
                nb_hidden2 = binary_step(nb_hidden2, False)
        
        if  len_h3:
            if used_h3 == nb_hidden3[0]:
                nb_hidden3 = binary_step(nb_hidden3, True)
            else:
                nb_hidden3 = binary_step(nb_hidden3, False)
        
        if len_log2_bs:
            if used_log2_bs == log2_batch_sizes[0]:
                log2_batch_sizes = binary_step(log2_batch_sizes, True)
            else:
                log2_batch_sizes = binary_step(log2_batch_sizes, False)

        if len_etas:
            if used_eta == etas[0]:
                etas = binary_step(etas, True)
            else:
                etas = binary_step(etas, False)

        if len_do:
            if used_do == dropout_probabilities[0]:
                dropout_probabilities = binary_step(dropout_probabilities, True)
            else:
                dropout_probabilities = binary_step(dropout_probabilities, False)

    return used_h1, used_h2, used_h3, used_do, used_log2_bs, used_eta 

def run_Net(cls, h1, h2, h3, do, log2_bs, eta, epochs, optim, alpha=1, beta=1, save_tensors=True):
    """
    train the network a number of times defined by NUMBER_OF_EVALUATION_RUNS and compute the avergare of the losses and accuracy
        cls:            class of the model
        h1, h2, h3:     number of hidden units 
        do:             dropout probability
        2^(log2_bs):    batch size 
        eta:            learning rate
        epochs:         number of epochs
        optim:          optimizer in {'SGD','Adam'}
        alpha, beta:    weight of the losses in models with auxiliary losses
    """
    filename = cls.__name__ + '_parameters.txt'
    
    with open(filename, "w") as f:
        f.write("h1: {}, h2: {}, h3: {}, do: {}, bs: 2**{}, eta: {}\n".format(h1, h2, h3, do, log2_bs, eta))
    
    filename = cls.__name__ + '_scores.txt'

    with open(filename, "w") as f:
        f.write("{} {}\n".format(epochs, NUMBER_OF_EVALUATION_RUNS))

    averaged_train_loss = 0
    averaged_test_loss = 0
    averaged_train_acc = 0
    averaged_test_acc = 0

    arr_train_losses = []
    arr_test_losses = []
    arr_train_accs = []
    arr_test_accs = []

    for i in range(NUMBER_OF_EVALUATION_RUNS):
        model = cls(h1, h2, h3, do)

        print('='*70)
        print('run: {:2d} - '.format(i+1) + model.get_str_parameters() + ', batch_size=2**{}, eta={:.4E}'.format(log2_bs,eta))
        print('-'*70)

        train_loader, test_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True)
        train_losses, test_losses, train_accs, test_accs  = train_Net(model=model, eta=eta, epochs=epochs, train_loader=train_loader, test_loader=test_loader,
                                                                             optim = optim, alpha = alpha, beta = beta)

        with open(filename, "a") as f:
            f.write(" ".join([str(l.item()) for l in train_losses])+"\n")
            f.write(" ".join([str(l.item()) for l in test_losses])+"\n")
            f.write(" ".join([str(er) for er in train_accs])+"\n")
            f.write(" ".join([str(er) for er in test_accs])+"\n")

        averaged_train_loss += train_losses[-1]
        averaged_test_loss += test_losses[-1]
        averaged_train_acc += train_accs[-1]
        averaged_test_acc += test_accs[-1]

        arr_train_losses.append(train_losses)
        arr_test_losses.append(test_losses)
        arr_train_accs.append(train_accs)
        arr_test_accs.append(test_accs)

    averaged_train_loss /= NUMBER_OF_EVALUATION_RUNS
    averaged_test_loss /= NUMBER_OF_EVALUATION_RUNS
    averaged_train_acc /= NUMBER_OF_EVALUATION_RUNS
    averaged_test_acc /= NUMBER_OF_EVALUATION_RUNS

    with open(filename, "a") as f:
        f.write("avg_train_loss: {:.4f}, avg_test_loss: {:.4f}, avg_train_acc: {:.4f}, avg_test_acc: {:.4f}\n".format(averaged_train_loss, averaged_test_loss, averaged_train_acc, averaged_test_acc))
        print("avg_train_loss: {:.4f}, avg_test_loss: {:.4f}, avg_train_acc: {:.4f}, avg_test_acc: {:.4f} saved to file: {}\n".format(averaged_train_loss, averaged_test_loss, averaged_train_acc, averaged_test_acc, filename))

    arr_train_losses = torch.tensor(arr_train_losses)
    arr_test_losses = torch.tensor(arr_test_losses)
    arr_train_accs = torch.tensor(arr_train_accs)
    arr_test_accs = torch.tensor(arr_test_accs)    

    if save_tensors:
        torch.save([arr_train_losses, arr_test_losses, arr_train_accs, arr_test_accs], '{}_tensors_to_plot.pt'.format(cls.__name__))

    return arr_train_losses, arr_test_losses, arr_train_accs, arr_test_accs



