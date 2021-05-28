import torch

from modules import *
from utils import *

import matplotlib.pyplot as plt


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

# Declare the model's architecture
model = Sequential(
    Linear(2, 25),ReLU(),
    Linear(25,25), ReLU(),
    Linear(25,1), Tanh(),
    MSELoss()
)

nb_train_samples = train_target.size(0)
nb_epochs = 300
batch_size = 50
eta = 1e-1

train_losses = []
test_losses = []
train_accs = []
test_accs = []

for e in range(nb_epochs):
    # Training mode
    model.train()

    for b in range(0, train_input.size(0), batch_size):

        prediction = model.forward(train_input.narrow(0, b, batch_size)) # inputs and outputs of linear layers are stored in Linear() objects
        loss = model.loss(prediction, train_target.narrow(0, b, batch_size)) # prediction and train_target are stored in Sequential() object  
        # print('epoch: {:4d}, batch: {:2d}, train_loss: {:8.4f} test_loss {:8.4f}'.format(e,  loss, test_loss)) 

        model.zero_grad() # gradients of linear layers are set to zero
        model.backward()# backpropagation - gradients are calculated and stored based on the prediction and target set in model.loss().
                        # backpropagation - the calculated gradients are stored as attributes in the object. 
        
        # update the parameters
        for p in model.with_parameters():
            p.weight = p.weight - eta * p.weight_grad
            p.bias = p.bias - eta * p.bias_grad
    
    #  Evaluation mode
    model.eval()

    train_losses.append(eval_model(model, train_input, train_target))
    test_losses.append(eval_model(model, test_input, test_target))
    train_accs.append(compute_acc(model, train_input, train_target))
    test_accs.append(compute_acc(model, test_input, test_target))

    if e%10 == 0:
        print('epoch: {:4d} - train_loss: {:8.4f} - test_loss {:8.4f} - train_acc: {:.4f} - test_cc {:.4f} '.format(e, train_losses[-1], test_losses[-1], train_accs[-1], test_accs[-1])) 
    
print('epoch: {:4d} - train_loss: {:8.4f} - test_loss {:8.4f} - train_acc: {:.4f} - test_cc {:.4f} '.format(e, train_losses[-1], test_losses[-1], train_accs[-1], test_accs[-1])) 

# convert lists to tensors
train_losses = torch.tensor(train_losses)
test_losses = torch.tensor(test_losses)
train_accs = torch.tensor(train_accs)
test_accs = torch.tensor(test_accs)

# uncomment to plot results - uncomment at the beggining import matplotlib.pyplot as plt
# plot_results(model, train_losses, test_losses, train_accs, test_accs, savefig=True)
# plt.show()
