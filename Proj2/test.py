import torch

from modules import *
from utils import *


torch.set_grad_enabled(False) # Set autograd off
torch.manual_seed(0) # Set manual seed for reproducibility
    
# Generate training and test set
train_input = torch.empty(1000,2).uniform_(0,1)
test_input = torch.empty(1000,2).uniform_(0,1)

train_target = is_inside(train_input)
test_target = is_inside(test_input)

# print(train_input.shape)
# print(test_input.shape)
# print(train_target.shape)
# print(test_target.shape)

model = Sequential(
    Linear(2, 25),Tanh(),
    Linear(25,25), Tanh(),
    Linear(25,2), ReLU(),
    MSELoss(reduction='sum')
)

nb_train_samples = train_target.size(0)
nb_epochs = 1000
# batch_size = 1000
nb_hidden = 50
eta = 1e-1 / nb_train_samples

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
            
