import torch
from torch import nn
from torch.nn import functional as F  

from data import *
from helpers import *
from time import time

BINARY_SEARCH_ITERATIONS = 4
NUMBER_OF_SEARCH_RUNS = 1
NUMBER_OF_EVALUATION_RUNS = 15
"""
[G_130521]
Note: 
    - binary search mods:
        - epoch is not a hyperparameter - e = 30  (speed up binary search)
        - introduce nb_hidden2 and nb_hidden3
        - max_pooling is always turned on
    - compute_error_rate moved from "helpers.py" to "aux_net.py"
ToDo:
    - try with Adam instead of SGD
    - scale the auxiliary loss
    - if accuracy is low try to remove max pooling
    - increase batch_size
    - consider increasing the depth and make the learning rate vary 
    - can be simplified with groups?
    - print computational time
"""
class AuxNet(nn.Module):
    def __init__(self, nb_hidden1, nb_hidden2, nb_hidden3, dropout_prob=0):
        super().__init__()
        self.feat_extractor1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), # 16x12x12    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), # 16x6x6
            nn.Conv2d(16, 32, kernel_size=3), # 32x4x4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), # 32x2x2
            nn.Conv2d(32, 64, kernel_size=2), # 64x1x1
            nn.ReLU()
        )
        self.feat_extractor2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), # 16x12x12    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), # 16x6x6
            nn.Conv2d(16, 32, kernel_size=3), # 32x4x4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), # 32x2x2
            nn.Conv2d(32, 64, kernel_size=2), # 64x1x1
            nn.ReLU()
        )
        self.classifier1 = nn.Sequential( 
            nn.Dropout(dropout_prob),
            nn.Linear(64, nb_hidden1), # nb_hidden1x1
            nn.BatchNorm1d(nb_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nb_hidden1, nb_hidden2), # nb_hidden2x1
            nn.ReLU(),
            nn.Linear(nb_hidden2,10) # 10x1
        )
        self.classifier2 = nn.Sequential( 
            nn.Dropout(dropout_prob),
            nn.Linear(64, nb_hidden1), # nb_hidden1x1
            nn.BatchNorm1d(nb_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nb_hidden1, nb_hidden2), # nb_hidden2x1
            nn.ReLU(),
            nn.Linear(nb_hidden2,10) # 10x1
        )
        self.final_classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(20, nb_hidden3), # nb_hidden3x1
            nn.BatchNorm1d(nb_hidden3),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(nb_hidden3, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
    
    def forward(self, x):
        # input separation
        x1 = x[:, 0, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))
        x2 = x[:, 1, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))
        # feature extraction
        x1 = self.feat_extractor1(x1)
        x2 = self.feat_extractor2(x2)
        # digit classification 
        x1 = self.classifier1(x1.view(-1, 64))
        x2 = self.classifier2(x2.view(-1, 64))
        # final classification
        y = torch.cat((x1, x2), 1)
        y = self.final_classifier(y)
        
        return x1, x2, y

def train_AuxNet(model, eta, train_loader, test_loader, eval_mode=False):
    losses = []
    train_error_rates = []
    test_error_rates = []
    device = get_device() 
    model  = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    epochs = 30   
    for _ in range(epochs):
        for input_data, target, class_data in iter(train_loader):
            output_aux1, output_aux2, output = model(input_data)
            # auxiliary losses
            loss_aux1 = criterion(output_aux1, class_data[:,0])
            loss_aux2 = criterion(output_aux2, class_data[:,1])
            # total losses
            loss = criterion(output, target) + loss_aux1 + loss_aux2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss)
        train_error_rates.append(compute_error_rate_AuxNet(model, train_loader))
        test_error_rates.append(compute_error_rate_AuxNet(model, test_loader))

    return losses, train_error_rates, test_error_rates

def compute_error_rate_AuxNet(model, data_loader):
    nb_errors = 0
    for input_data, target, _ in iter(data_loader):
        _, _, output = model(input_data)
        for i, out in enumerate(output):
            pred_target = out.max(0)[1].item()
            if (target[i]) != pred_target:
                nb_errors += 1

    error_rate = nb_errors/len(data_loader.dataset)

    return error_rate

def compute_center(two_elements_list):
    return (two_elements_list[0]+two_elements_list[1])/2

def binary_step(two_elements_list, left):
    if left:
        return [two_elements_list[0], compute_center(two_elements_list)]
    else:
        return [compute_center(two_elements_list), two_elements_list[1]]

def binary_search_AuxNet(hidden_layers1, hidden_layers2, hidden_layers3, batch_sizes, etas, dropout_probabilities):
    lowest_error_rate = float('inf')
    
    used_hl = -1
    used_h2 = -1
    used_h3 = -1
    used_bs = -1
    used_eta = -1
    used_do = -1
    filename = "AuxNet_binarysearch.txt"
    f = open(filename, "w")
    for bsi in range(BINARY_SEARCH_ITERATIONS):
        assert(len(hidden_layers1) == 2)
        assert(len(hidden_layers2) == 2)
        assert(len(hidden_layers3) == 2)
        assert(len(batch_sizes) == 2)
        assert(len(etas) == 2)
        assert(len(dropout_probabilities) == 2)

        hidden_layers1 = [int(hl) for hl in hidden_layers1]
        hidden_layers2 = [int(h2) for h2 in hidden_layers2]
        hidden_layers3 = [int(h3) for h3 in hidden_layers3]    
        batch_sizes = [int(bs) for bs in batch_sizes]

        for hl in hidden_layers1:
            for h2 in hidden_layers2:
                for h3 in hidden_layers3:
                    for bs in batch_sizes:
                        for eta in etas:
                            for do in dropout_probabilities:
                                last_time = time()
                                error_rate = 0
                                for r in range(NUMBER_OF_SEARCH_RUNS):
                                    model = AuxNet(hl, h2, h3, do)
                                    train_loader, test_loader = get_data(N=1000, batch_size=2**bs, shuffle=True)
                                    _, _, er = train_AuxNet(model, eta, train_loader, test_loader)
                                    error_rate += er[-1]
                                    del model
                                averaged_error_rate = error_rate/NUMBER_OF_SEARCH_RUNS
                                if averaged_error_rate < lowest_error_rate:
                                    lowest_error_rate = averaged_error_rate
                                    used_hl = hl
                                    used_h2 = h2
                                    used_h3 = h3
                                    used_bs = bs
                                    used_eta = eta
                                    used_do = do

                                f.write("bsi: {:1.0f}, hl: {:4.0f}, h2: {:4.0f}, h3: {:4.0f}, bs: 2**{:1.0f}, eta: {:.5f}, do: {:.5f} -> er: {:.3f} in about {:.2f}sec".format(bsi, hl, h2, h3, bs, eta, do, averaged_error_rate, time()-last_time))

        if used_hl == hidden_layers1[0]:
            hidden_layers1 = binary_step(hidden_layers1, True)
        else:
            hidden_layers1 = binary_step(hidden_layers1, False)

        if used_h2 == hidden_layers2[0]:
            hidden_layers2 = binary_step(hidden_layers2, True)
        else:
            hidden_layers2 = binary_step(hidden_layers2, False)

        if used_h3 == hidden_layers3[0]:
            hidden_layers3 = binary_step(hidden_layers3, True)
        else:
            hidden_layers3 = binary_step(hidden_layers3, False)  

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
    f.colse()
    return used_hl, used_h2, used_h3, used_bs, used_eta, used_do 


def eval_AuxNet():
    hidden_layers1 = [10, 1000]
    hidden_layers2 = [10, 1000]
    hidden_layers3 = [10, 1000]
    batch_sizes = [3, 7]
    etas = [1e-3, 1e-1]
    dropout_probabilities = [0, 0.9]
    epochs = 30

    hl, h2, h3, bs, eta, do = binary_search_AuxNet(hidden_layers1, hidden_layers2, hidden_layers3, batch_sizes, etas, dropout_probabilities)

    filename = "AuxNet_parameters.txt"
    f = open(filename, "w")
    f.write("hl: {}, h2: {}, h3: {}, bs: 2**{}, eta: {}, do: {}\n".format(hl, h2, h3, bs, eta, do))
    f.close()

    filename = "AuxNet_scores.txt"

    f = open(filename, "w")
    f.write("{} {}\n".format(epochs, NUMBER_OF_EVALUATION_RUNS))

    averaged_losses = 0
    averaged_train_error_rate = 0
    averaged_test_error_rate = 0

    for i in range(NUMBER_OF_EVALUATION_RUNS):
        model = AuxNet(hl, h2, h3, do)
        train_loader, test_loader = get_data(N=1000, batch_size=2**bs, shuffle=True)
        losses, train_error_rates, test_error_rates = train_AuxNet(model, eta, train_loader, test_loader)

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

    f.write("[average] loss: {}, train error rate: {}, test error rate: {}\n".format(averaged_losses, averaged_train_error_rate, averaged_test_error_rate))
    # f.write('[min] loss: {}, train error rate: {}, test_error_rate: {}\n'.)
    f.close()

def run_AuxNet(hl, h2, h3, bs, eta, do):
    averaged_time = 0
    averaged_losses = 0
    averaged_train_error_rate = 0
    averaged_test_error_rate = 0

    for i in range(NUMBER_OF_EVALUATION_RUNS):
        start_time = time()
        model = AuxNet(hl, h2, h3, do)
        train_loader, test_loader = get_data(N=1000, batch_size=2**bs, shuffle=True)
        losses, train_error_rates, test_error_rates = train_AuxNet(model, eta, train_loader, test_loader)
        averaged_time += time() - start_time
        averaged_losses += losses[-1]
        averaged_train_error_rate += train_error_rates[-1]
        averaged_test_error_rate += test_error_rates[-1]
        print('evaluation run {}'.format(i))
        del model

    averaged_time /= NUMBER_OF_EVALUATION_RUNS
    averaged_losses /= NUMBER_OF_EVALUATION_RUNS
    averaged_train_error_rate /= NUMBER_OF_EVALUATION_RUNS
    averaged_test_error_rate /= NUMBER_OF_EVALUATION_RUNS

    print("loss: {:.5f}, train error rate: {:.5f}, test error rate: {:.5f}, average time {:.5f} \n".format(averaged_losses, averaged_train_error_rate, averaged_test_error_rate, averaged_time))


