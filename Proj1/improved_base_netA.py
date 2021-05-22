import torch
from torch import nn
from torch.nn import functional as F  

from data import *
from helpers import *
from time import time
import os

BINARY_SEARCH_ITERATIONS = 4
NUMBER_OF_SEARCH_RUNS = 1
NUMBER_OF_EVALUATION_RUNS = 15

class BaseNetCNN(nn.Module):
    def __init__(self, nb_hidden1, nb_hidden2, dropout_prob=0):
        super().__init__()

        self.nb_hidden1 = nb_hidden1
        self.nb_hidden2 = nb_hidden2
        self.dropout_prob = dropout_prob
        
        # fully convoluted
        self.conv1 = nn.Conv2d( 2, 32, kernel_size=5) 
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5) 
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3) 
        self.batchNorm3 = nn.BatchNorm2d(64)
        
        # fully connected 
        self.fc1 = nn.Linear(256, self.nb_hidden1)
        self.batchNorm4 = nn.BatchNorm1d(self.nb_hidden1)
        self.fc2 = nn.Linear(self.nb_hidden1, self.nb_hidden2)
        self.batchNorm5 = nn.BatchNorm1d(self.nb_hidden2)
        self.fc3 = nn.Linear(self.nb_hidden2, 2)

        self.max_pool = nn.MaxPool2d(kernel_size = 2)
        self.dropout = nn.Dropout(self.dropout_prob)
        # batch norms

        self.relu = nn.ReLU()

    def forward(self, x): 
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.conv3(x)                                       # 64x4x4
        x = self.batchNorm3(x)                                  # 64x4x4
        x = self.relu(x)                                        # 64x4x4
        x = self.max_pool(x)                                    # 64x2x2
        x = self.fc1(x.view(-1, 256))                           # nb_hidden1x1
        x = self.batchNorm4(x)                                  # nb_hidden1x1
        x = self.relu(x)                                        # nb_hidden1x1
        x = self.dropout(x)                                     # nb_hidden1x1
        x = self.fc2(x)                                         # nb_hidden2x1
        x = self.batchNorm5(x)                                  # nb_hidden2x1
        x = self.relu(x)                                        # nb_hidden2x1
        x = self.dropout(x)                                     # nb_hidden2x1
        x = self.fc3(x)                                         # 2x1
        return x
    
    def get_str_parameters(self):
        return '{} - nb_hidden1={}, nb_hidden2={}, dropout={}'.format(
                type(self).__name__, self.nb_hidden1, self.nb_hidden2, self.dropout_prob)

class BaseNetMLP(nn.Module):
    def __init__(self, nb_hidden1, nb_hidden2, dropout_prob=0):
        super().__init__()

        self.nb_hidden1 = nb_hidden1
        self.nb_hidden2 = nb_hidden2
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(2*14*14, self.nb_hidden1)
        self.batchNorm1 = nn.BatchNorm1d(self.nb_hidden1)
        self.fc2 = nn.Linear(self.nb_hidden1, self.nb_hidden2)
        self.batchNorm2 = nn.BatchNorm1d(self.nb_hidden2)
        self.fc3 = nn.Linear(self.nb_hidden2, 2)


        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x.view(-1, 2*14*14))
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
    
    def get_str_parameters(self):
        return '{} - nb_hidden1={}, nb_hidden2={}, dropout={}'.format(
                type(self).__name__, self.nb_hidden1, self.nb_hidden2, self.dropout_prob)


def train_BaseNet(model, eta, epochs, train_loader, test_loader, optim = 'SGD', print_results=True):
    model.train() # set training mode for BatchNorm and Dropout
    
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
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)
        
    for e in range(epochs):
        model.train()
        for input_data, target, _ in iter(train_loader):
            output = model(input_data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(eval_model(model=model, data_loader=train_loader))
        test_losses.append(eval_model(model=model, data_loader=test_loader))
        train_accs.append(compute_acc(model=model,data_loader=train_loader)) # model.eval() and torch.no_grad() is used while evaluating
        test_accs.append(compute_acc(model=model,data_loader=test_loader)) # model.eval() and torch.no_grad() is used while evaluating
         
        if print_results:
            if ((e%10) == 0):
                print(get_str_results(epoch=e, train_loss= train_losses[-1], test_loss=test_losses[-1] , train_acc= train_accs[-1], test_acc=test_accs[-1]))

    if print_results:           
        print(get_str_results(epoch=e, train_loss= train_losses[-1], test_loss=test_losses[-1] , train_acc= train_accs[-1], test_acc=test_accs[-1]))
        print(70*'-')

    return train_losses, test_losses, train_accs, test_accs


def compute_center(two_elements_list):
    return (two_elements_list[0]+two_elements_list[1])/2

def binary_step(two_elements_list, left):
    if left:
        return [two_elements_list[0], compute_center(two_elements_list)]
    else:
        return [compute_center(two_elements_list), two_elements_list[1]]

def binary_search_BaseNet(hidden_layers1, hidden_layers2, dropout_probabilities, log2_batch_sizes, etas, epochs, cls=BaseNetCNN):
    highest_acc = float('-inf')

    used_hl = -1
    used_h2 = -1
    used_log2_bs = -1
    used_eta = -1
    used_do = -1

    if cls == BaseNetCNN:
        filename = "./BaseNetCNN"
    elif cls == BaseNetMLP:
        filename = "./BaseNetMLP"
    filename += "_binarysearch.txt"

    if os.path.exists(filename):
        os.remove(filename)
    
    for bsi in range(BINARY_SEARCH_ITERATIONS):

        len_hl = len(hidden_layers1) == 2
        len_h2 = len(hidden_layers2) == 2
        len_log2_bs = len(log2_batch_sizes) == 2
        len_etas = len(etas) == 2
        len_do = len(dropout_probabilities) == 2

        hidden_layers1 = [int(hl) for hl in hidden_layers1]
        hidden_layers2 = [int(h2) for h2 in hidden_layers2]
        log2_batch_sizes = [int(log2_bs) for log2_bs in log2_batch_sizes]

        for hl in hidden_layers1:
            for h2 in hidden_layers2:
                for log2_bs in log2_batch_sizes:
                     for eta in etas:
                        for do in dropout_probabilities:
                            last_time = time()
                            acc_cum = 0
                            for r in range(NUMBER_OF_SEARCH_RUNS):
                                model = cls(hl, h2, do) # By default cls = SiamesetNet
                                train_loader, test_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True, validation = True, val_size = 800)
                                _, _, _, test_accs = train_BaseNet(model=model, eta=eta, epochs=epochs, train_loader=train_loader, test_loader=test_loader)
                                # print(get_str_results(epoch=e, train_loss= train_losses[-1], test_loss=test_losses[-1] , train_acc= train_accs[-1], test_acc=test_accs[-1]))
                                acc_cum += test_accs[-1]
                                del model
                            averaged_acc = acc_cum/NUMBER_OF_SEARCH_RUNS
                            print(averaged_acc , highest_acc)   
                            if averaged_acc > highest_acc:
                                highest_acc = averaged_acc
                                used_hl = hl
                                used_h2 = h2
                                used_log2_bs = log2_bs
                                used_eta = eta
                                used_do = do

                            # print('-'*70)
                            print("bsi: {:2d}, hl: {}, h2: {}, do: {:.3f}, bs: 2**{}, eta: {:.4E} -> er: {:.4f} in about {:.2f}sec".format(bsi, hl, h2, do, log2_bs, eta, averaged_acc, time()-last_time))
                            print('='*70)
                            # print('='*70 +'\n' + '='*70)
                            with open(filename, "a") as f:
                                f.write("bsi: {:2d}, hl: {}, h2: {}, do: {:.3f}, bs: 2**{}, eta: {:.4E} -> er: {:.4f} in about {:.2f}sec\n".format(bsi, hl, h2, do, log2_bs, eta, averaged_acc, time()-last_time))
        
        if not (len_hl or len_h2 or len_log2_bs or len_etas or len_do): # if binary search is not possible -> break the loop
            break

        if len_hl:
            if used_hl == hidden_layers1[0]:
                hidden_layers1 = binary_step(hidden_layers1, True)
            else:
                hidden_layers1 = binary_step(hidden_layers1, False)
        
        if  len_h2:
            if used_h2 == hidden_layers2[0]:
                hidden_layers2 = binary_step(hidden_layers2, True)
            else:
                hidden_layers2 = binary_step(hidden_layers2, False)
        
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

    return used_hl, used_h2, used_do, used_log2_bs, used_eta 


def run_BaseNet(hl, h2, do, log2_bs, eta, epochs, save_tensors=True, cls=BaseNetCNN):

    if cls == BaseNetCNN:
        filename = "./BaseNetCNN"
    elif cls == BaseNetMLP:
        filename = "./BaseNetMLP"
    filename += "_parameters.txt"

    with open(filename, "w") as f:
        f.write("hl: {}, h2: {}, do: {}, bs: 2**{}, eta: {}, \n".format(hl, h2, do, log2_bs, eta))
    
    if cls == BaseNetCNN:
        filename = "./BaseNetCNN"
    elif cls == BaseNetMLP:
        filename = "./BaseNetMLP"
    filename += "_scores.txt"

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
        model = cls(hl, h2, do)

        print('='*70)
        print('run: {:2d} - '.format(i) + model.get_str_parameters() + ', batch_size=2**{}, eta={:.4E}'.format(log2_bs,eta))
        print('-'*70)
        
        train_loader, test_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True)
        train_losses, test_losses, train_accs, test_accs  = train_BaseNet(model=model, eta=eta, epochs=epochs, train_loader=train_loader, test_loader=test_loader)
        # print(get_str_results(epoch=epochs-1, train_loss=train_losses[-1], test_loss=test_losses[-1], train_acc=train_accs[-1], test_acc=test_accs[-1]))
 
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

        del model

    averaged_train_loss /= NUMBER_OF_EVALUATION_RUNS
    averaged_test_loss /= NUMBER_OF_EVALUATION_RUNS
    averaged_train_acc /= NUMBER_OF_EVALUATION_RUNS
    averaged_test_acc /= NUMBER_OF_EVALUATION_RUNS

    with open(filename, "a") as f:
        f.write("avg_train_loss: {:.4f}, avg_test_loss: {:.4f} ,avg_train_error: {:.4f}, avg_test_error: {:.4f}\n".format(averaged_train_loss, averaged_test_loss, averaged_train_acc, averaged_test_acc))
        print("avg_train_loss: {:.4f}, avg_test_loss: {:.4f} ,avg_train_error: {:.4f}, avg_test_error: {:.4f} saved to file: {}\n".format(averaged_train_loss, averaged_test_loss, averaged_train_acc, averaged_test_acc, filename))

    arr_train_losses = torch.tensor(arr_train_losses)
    arr_test_losses = torch.tensor(arr_test_losses)
    arr_train_accs = torch.tensor(arr_train_accs)
    arr_test_accs = torch.tensor(arr_test_accs)    

    if save_tensors:
        torch.save([arr_train_losses, arr_test_losses, arr_train_accs, arr_test_accs], '{}_tensors_to_plot.pt'.format('SiameseNet'))

    return arr_train_losses, arr_test_losses, arr_train_accs, arr_test_accs