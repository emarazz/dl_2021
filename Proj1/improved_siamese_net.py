import torch
from torch import nn
from torch.nn import functional as F  

from data import *
from helpers import *
from time import time
import os

BINARY_SEARCH_ITERATIONS = 4
NUMBER_OF_SEARCH_RUNS = 1
NUMBER_OF_EVALUATION_RUNS = 2
"""
siamese net based on improved AuxNet
"""
class SiameseNet(nn.Module):
    def __init__(self, nb_hidden1, nb_hidden2, nb_hidden3, dropout_prob=0):
        super().__init__()

        self.nb_hidden1 = nb_hidden1
        self.nb_hidden2 = nb_hidden2
        self.nb_hidden3 = nb_hidden3
        self.dropout_prob = dropout_prob        

        self.feat_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), # 16x12x12    
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3), # 32x10x10
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), # 64x8x8
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), # 64x4x4
            nn.Conv2d(64, 64, kernel_size=3), # 64x2x2
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )
        self.classifier = nn.Sequential( 
            nn.Dropout(dropout_prob),
            nn.Linear(256, nb_hidden1), # nb_hidden1x1
            nn.BatchNorm1d(nb_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nb_hidden1, nb_hidden2), # nb_hidden2x1
            nn.BatchNorm1d(nb_hidden2),
            nn.ReLU(),
            nn.Linear(nb_hidden2,10) # 10x1
        )
        self.final_classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(20, nb_hidden3), # nb_hidden3x1
            nn.BatchNorm1d(nb_hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nb_hidden3, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
    
    def forward(self, x):
        # input separation
        x1 = x[:, 0, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))
        x2 = x[:, 1, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))
        # feature extraction
        x1 = self.feat_extractor(x1)
        # digit classification 
        x2 = self.feat_extractor(x2)
        x1 = self.classifier(x1.view(-1, 256))
        x2 = self.classifier(x2.view(-1, 256))
        # final classification
        y = torch.cat((x1, x2), 1)
        y = self.final_classifier(y)
        
        return x1, x2, y

    def get_str_parameters(self):
        return '{} - nb_hidden1={}, nb_hidden2={}, nb_hidden3={}, dropout={}'.format(
                type(self).__name__, self.nb_hidden1, self.nb_hidden2, self.nb_hidden3, self.dropout_prob)


class CNN_AUX_WS(nn.Module):
    name = '3 Layer Convolution, weight sharing with auxiliary Loss'
    name_prefix = 'CNN+AUX+WS'

    def __init__(self, nb_hidden1, nb_hidden2, nb_hidden3, dropout_prob=0):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(CNN_AUX_WS, self).__init__()

        self.nb_hidden1 = nb_hidden1
        self.nb_hidden2 = nb_hidden2
        self.nb_hidden3 = nb_hidden3
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # 14 -> 12
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 12 -> 10
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)  # 10 -> 8
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.fc1 = nn.Conv2d(64, 32 , kernel_size = 3)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.fc2 = nn.Conv2d(32,16, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(num_features=16)
        self.fc3 = nn.Conv2d(16, 10, kernel_size=4) # Here it ends the number classifier

        self.bn6 = nn.BatchNorm1d(num_features=20)         
        self.lin1 = nn.Linear(20, 256)
        self.bn7 = nn.BatchNorm1d(num_features=256)
        self.lin2 = nn.Linear(256, 256)
        self.bn8 = nn.BatchNorm1d(num_features=256)
        self.lin3 = nn.Linear(256, 2)

    def forward(self, x):

        xA = x[:, 0].unsqueeze(1)
        xB = x[:, 1].unsqueeze(1)

        number0 = F.relu(self.conv1(xA))
        number0 = F.relu(self.conv2(self.bn1(number0)))
        number0 = F.relu(self.conv3(self.bn2(number0)))
        number0 = F.relu(self.fc1(self.bn3(number0)))
        number0 = F.relu(self.fc2(self.bn4(number0)))
        number0 = torch.tanh(self.fc3(self.bn5(number0)))
        number0 = number0.view(-1,10)
        
        number1 = F.relu(self.conv1(xB))
        number1 = F.relu(self.conv2(self.bn1(number1)))
        number1 = F.relu(self.conv3(self.bn2(number1)))
        number1 = F.relu(self.fc1(self.bn3(number1)))
        number1 = F.relu(self.fc2(self.bn4(number1)))
        number1 = torch.tanh(self.fc3(self.bn5(number1)))
        number1 = number1.view(-1,10)
        
        minus = F.relu(self.lin1(self.bn6(torch.cat((number0, number1), 1))))
        minus = F.relu(self.lin2(self.bn7(minus)))
        minus = torch.sigmoid(self.lin3(self.bn8(minus)))

        return number0, number1, minus
    
    def get_str_parameters(self):
        return '{} - nb_hidden1={}, nb_hidden2={}, nb_hidden3={}, dropout={}'.format(
                type(self).__name__, self.nb_hidden1, self.nb_hidden2, self.nb_hidden3, self.dropout_prob)


def train_SiameseNet(model, eta, epochs, train_loader, val_loader, optim = 'Adam', print_results=True):
    model.train()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
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
        # print(model.training)
        
        for input_data, target, class_data in iter(train_loader):
            output_aux1, output_aux2, output = model(input_data)
            # auxiliary losses
            loss_aux1 = criterion(output_aux1, class_data[:,0])
            loss_aux2 = criterion(output_aux2, class_data[:,1])
            # total losses
            # loss = criterion(output, target) + 0.75*loss_aux1 + 0.75*loss_aux2
            lam = 0.5
            loss = lam * criterion(output, target) + loss_aux1 + loss_aux2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(eval_model(model=model, data_loader=train_loader))
        val_losses.append(eval_model(model=model, data_loader=val_loader))
        train_accs.append(compute_acc(model=model,data_loader=train_loader)) # model.eval() and torch.no_grad() is used while evaluating
        val_accs.append(compute_acc(model=model,data_loader=val_loader)) # model.eval() and torch.no_grad() is used while evaluating
        
        if print_results:
            if ((e%10) == 0):
                print(get_str_results(epoch=e, train_loss= train_losses[-1], val_loss=val_losses[-1] , train_acc= train_accs[-1], val_acc=val_accs[-1]))

    if print_results:           
        print(get_str_results(epoch=e, train_loss= train_losses[-1], val_loss=val_losses[-1] , train_acc= train_accs[-1], val_acc=val_accs[-1]))
        print(70*'-')

    return train_losses, val_losses, train_accs, val_accs

# @torch.no_grad()
# def eval_SiameseNet(model, epochs, val_loader, print_results=True):
#     model.eval()
    
#     val_losses = []
#     val_acc_rates = []
    
#     device = get_device() 
#     model  = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     criterion = criterion.to(device)

#     for e in range(epochs):
#         for input_data, target, class_data in iter(val_loader):
#             output_aux1, output_aux2, output = model(input_data)
#             # auxiliary losses
#             loss_aux1 = criterion(output_aux1, class_data[:,0])
#             loss_aux2 = criterion(output_aux2, class_data[:,1])
#             # total losses
#             loss = criterion(output, target) + 0.75*loss_aux1 + 0.75*loss_aux2

#         val_losses.append(loss)
#         val_acc_rates.append(compute_acc_rate(output=output, target=target)) # model.eval() and torch.no_grad() is used while evaluating
        
#         if print_results:
#             if ((e%10) == 0):
#                 print(get_str_results(epoch=e, val_loss= val_losses[-1], val_acc_rate= val_acc_rates[-1]))#, val_acc_rates[-1]))

#     if print_results:           
#         print(get_str_results(epoch=e, val_loss= val_losses[-1], val_acc_rate= val_acc_rates[-1]))#, val_acc_rates[-1]))  
#         print(70*'-')

#     return val_losses, val_acc_rates


def compute_center(two_elements_list):
    return (two_elements_list[0]+two_elements_list[1])/2

def binary_step(two_elements_list, left):
    if left:
        return [two_elements_list[0], compute_center(two_elements_list)]
    else:
        return [compute_center(two_elements_list), two_elements_list[1]]

def binary_search_SiameseNet(hidden_layers1, hidden_layers2, hidden_layers3, dropout_probabilities, log2_batch_sizes, etas, epochs, cls=SiameseNet):
    highest_acc = float('-inf')
    
    used_hl = -1
    used_h2 = -1
    used_h3 = -1
    used_log2_bs = -1
    used_eta = -1
    used_do = -1

    filename = "SiameseNet_binarysearch.txt"
    if os.path.exists(filename):
        os.remove(filename)

    for bsi in range(BINARY_SEARCH_ITERATIONS):
        # assert(len(hidden_layers1) == 2)
        # assert(len(hidden_layers2) == 2)
        # assert(len(hidden_layers3) == 2)
        # assert(len(log2_batch_sizes) == 2)
        # assert(len(etas) == 2)
        # assert(len(dropout_probabilities) == 2)
        
        len_hl = len(hidden_layers1) == 2
        len_h2 = len(hidden_layers2) == 2
        len_h3 = len(hidden_layers2) == 2
        len_log2_bs = len(log2_batch_sizes) == 2
        len_etas = len(etas) == 2
        len_do = len(dropout_probabilities) == 2

        hidden_layers1 = [int(hl) for hl in hidden_layers1]
        hidden_layers2 = [int(h2) for h2 in hidden_layers2]
        hidden_layers3 = [int(h3) for h3 in hidden_layers3]    
        log2_batch_sizes = [int(log2_bs) for log2_bs in log2_batch_sizes]

        for hl in hidden_layers1:
            for h2 in hidden_layers2:
                for h3 in hidden_layers3:
                    for log2_bs in log2_batch_sizes:
                        for eta in etas:
                            for do in dropout_probabilities:
                                last_time = time()
                                acc_cum = 0
                                for r in range(NUMBER_OF_SEARCH_RUNS):
                                    model = cls(hl, h2, h3, do) # By default cls = SiamesetNet
                                    train_loader, val_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True)
                                    _, _, train_accs, _ = train_SiameseNet(model=model, eta=eta, epochs=epochs, train_loader=train_loader, val_loader=val_loader)
                                    # print(get_str_results(epoch=e, train_loss= train_losses[-1], val_loss=val_losses[-1] , train_acc= train_accs[-1], val_acc=val_accs[-1]))
                                    acc_cum += train_accs[-1]
                                    del model
                                averaged_acc = acc_cum/NUMBER_OF_SEARCH_RUNS
                                if averaged_acc > highest_acc:
                                    highest_acc = highest_acc
                                    used_hl = hl
                                    used_h2 = h2
                                    used_h3 = h3
                                    used_log2_bs = log2_bs
                                    used_eta = eta
                                    used_do = do
                                
                                print('-'*70)
                                print("bsi: {:2d}, hl: {}, h2: {}, do: {:.3f}, bs: 2**{}, eta: {:.4E} -> er: {:.4f} in about {:.2f}sec".format(bsi, hl, h2, do, log2_bs, eta, averaged_acc, time()-last_time))
                                print('='*70 +'\n' + '='*70)
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
        
        if  len_h3:
            if used_h3 == hidden_layers3[0]:
                hidden_layers3 = binary_step(hidden_layers3, True)
            else:
                hidden_layers3 = binary_step(hidden_layers3, False)
        
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

    return used_hl, used_h2, used_h3, used_do, used_log2_bs, used_eta 


def run_SiameseNet(hl, h2, h3, do, log2_bs, eta, epochs, save_tensors=True, cls=SiameseNet):
    
    filename = "SiameseNet_parameters.txt"
    
    with open(filename, "w") as f:
        f.write("hl: {}, h2: {}, h3: {}, do: {}, bs: 2**{}, eta: {}\n".format(hl, h2, h3, do, log2_bs, eta))
    
    filename = "SiameseNet_scores.txt"

    with open(filename, "w") as f:
        f.write("{} {}\n".format(epochs, NUMBER_OF_EVALUATION_RUNS))

    averaged_train_loss = 0
    averaged_val_loss = 0
    averaged_train_acc = 0
    averaged_val_acc = 0

    arr_train_losses = []
    arr_val_losses = []
    arr_train_accs = []
    arr_val_accs = []

    for i in range(NUMBER_OF_EVALUATION_RUNS):
        model = cls(hl, h2, h3, do)

        print('='*70)
        print('run: {:2d} - '.format(i) + model.get_str_parameters() + ', batch_size=2**{}, eta={:.4E}'.format(log2_bs,eta))
        print('-'*70)

        train_loader, val_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True)
        train_losses, val_losses, train_accs, val_accs  = train_SiameseNet(model=model, eta=eta, epochs=epochs, train_loader=train_loader, val_loader=val_loader)
        # print(get_str_results(epoch=epochs-1, train_loss=train_losses[-1], val_loss=val_losses[-1], train_acc=train_accs[-1], val_acc=val_accs[-1]))

        with open(filename, "a") as f:
            f.write(" ".join([str(l.item()) for l in train_losses])+"\n")
            f.write(" ".join([str(l.item()) for l in val_losses])+"\n")
            f.write(" ".join([str(er) for er in train_accs])+"\n")
            f.write(" ".join([str(er) for er in val_accs])+"\n")

        averaged_train_loss += train_losses[-1]
        averaged_val_loss += val_losses[-1]
        averaged_train_acc += train_accs[-1]
        averaged_val_acc += val_accs[-1]

        arr_train_losses.append(train_losses)
        arr_val_losses.append(val_losses)
        arr_train_accs.append(train_accs)
        arr_val_accs.append(val_accs)

        del model

    averaged_train_loss /= NUMBER_OF_EVALUATION_RUNS
    averaged_val_loss /= NUMBER_OF_EVALUATION_RUNS
    averaged_train_acc /= NUMBER_OF_EVALUATION_RUNS
    averaged_val_acc /= NUMBER_OF_EVALUATION_RUNS

    with open(filename, "a") as f:
        f.write("avg_train_loss: {:.4f}, avg_val_loss: {:.4f} ,avg_train_error: {:.4f}, avg_val_error: {:.4f}\n".format(averaged_train_loss, averaged_val_loss, averaged_train_acc, averaged_val_acc))
        print("avg_train_loss: {:.4f}, avg_val_loss: {:.4f} ,avg_train_error: {:.4f}, avg_val_error: {:.4f} saved to file: {}\n".format(averaged_train_loss, averaged_val_loss, averaged_train_acc, averaged_val_acc, filename))

    arr_train_losses = torch.tensor(arr_train_losses)
    arr_val_losses = torch.tensor(arr_val_losses)
    arr_train_accs = torch.tensor(arr_train_accs)
    arr_val_accs = torch.tensor(arr_val_accs)    

    if save_tensors:
        torch.save([arr_train_losses, arr_val_losses, arr_train_accs, arr_val_accs], '{}_tensors_to_plot.pt'.format(SiameseNet))

    return arr_train_losses, arr_val_losses, arr_train_accs, arr_val_accs

# def run_SiameseNet(hl, h2, h3, bs, eta, do):
#     averaged_losses = 0
#     averaged_train_acc = 0
#     averaged_val_acc = 0

#     for i in range(NUMBER_OF_EVALUATION_RUNS):
#         model = SiameseNet(hl, h2, h3, do)
#         train_loader, val_loader = get_data(N=1000, batch_size=2**bs, shuffle=True)
#         losses, train_acc_rates, val_acc_rates = train_SiameseNet(model, eta, train_loader, val_loader)

#         averaged_losses += losses[-1]
#         averaged_train_acc += train_acc_rates[-1]
#         averaged_val_acc += val_acc_rates[-1]

#         del model

#     averaged_losses /= NUMBER_OF_EVALUATION_RUNS
#     averaged_train_acc /= NUMBER_OF_EVALUATION_RUNS
#     averaged_val_acc /= NUMBER_OF_EVALUATION_RUNS

#     print("loss: {}, train error rate: {}, test error rate: {} saved to file\n".format(averaged_losses, averaged_train_acc, averaged_val_acc))
