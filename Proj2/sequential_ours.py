import torch
from torch import Tensor

import dlc_practical_prologue as prologue
from modules import *

train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels = True,
                                                                        normalize = True)
nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)

zeta = 0.90

train_target = train_target * zeta
test_target = test_target * zeta


torch.manual_seed(0)


model = Sequential(
    Linear(784, 100),Tanh(),
    Linear(100,100), Tanh(),
    Linear(100,100), Tanh(),
    Linear(100,10), ReLU(),
    MSELoss(reduction='sum')
)

nb_epochs = 1000
batch_size = 1000
nb_hidden = 50
eta = 1e-1 / nb_train_samples
epsilon = 1e-6

for e in range(nb_epochs):

    #### test predictions should be placed CAREFULLY because it stores values!!!
    test_prediction = model.forward(test_input) 
    test_loss = model.loss(test_prediction, test_target)
    ####

    prediction = model.forward(train_input) # inputs and outputs of linear layers are stored in Lines() objects
    loss = model.loss(prediction, train_target) # prediction and train_target are stored in Sequential() object  
    print('epoch: {:4d} train_loss: {:8.4f} test_loss {:8.4f}'.format(e+1, loss, test_loss)) 

    model.zero_grad() # gradeints of linear layers are set to zero
    model.backward()# backpropagation - gradients are calculated and stored based on the prediction and target set in model.loss().
                    # backpropagation - the calculated gradients are stored as attributes in the object. 

    for p in model.with_parameters():
        p.weight = p.weight - eta * p.weight_grad
        p.bias = p.bias - eta * p.bias_grad
            
