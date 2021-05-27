import torch
import torch.nn as nn

from utils_pytorch import *

import matplotlib.pyplot as plt


# torch.set_grad_enabled(False) # Set autograd off
torch.manual_seed(0) # Set manual seed for reproducibility
    
# Generate training and test set
train_input = torch.empty(1000,2).uniform_(0,1)
test_input = torch.empty(1000,2).uniform_(0,1)

train_target = is_inside2(train_input).float()
test_target = is_inside2(test_input).float()

# print(train_input.shape)
# print(test_input.shape)
# print(train_target.shape)
# print(test_target.shape)

# Declare the model's architecture
model = nn.Sequential(
    nn.Linear(2, 25),nn.ReLU(),
    nn.Linear(25,25), nn.ReLU(),
    nn.Linear(25,1), nn.Tanh()
)

nb_train_samples = train_target.size(0)
nb_epochs = 300
batch_size = 50
eta = 1e-1

train_losses = []
test_losses = []
train_accs = []
test_accs = []

criterion = nn.MSELoss()

for e in range(nb_epochs):
    # Training mode
    model.train()
    for b in range(0, train_input.size(0), batch_size):

        prediction = model(train_input.narrow(0, b, batch_size)) # inputs and outputs of linear layers are stored in Linear() objects
        loss = criterion(prediction, train_target.narrow(0, b, batch_size)) # prediction and train_target are stored in Sequential() object  
        # print(loss)
        model.zero_grad() 
        loss.backward() 

        with torch.no_grad():
            for p in model.parameters():
                p -=  eta * p.grad
        # print(e, loss)

    #  Evaluation mode
    model.eval()

    train_losses.append(eval_model_pytorch(model, train_input, train_target))
    test_losses.append(eval_model_pytorch(model, test_input, test_target))
    train_accs.append(compute_acc_pytorch(model, train_input, train_target))
    test_accs.append(compute_acc_pytorch(model, test_input, test_target))

    if e%10 == 0:
        print('epoch: {:4d} - train_loss: {:8.4f} - test_loss {:8.4f} - train_acc: {:.4f} - test_cc {:.4f} '.format(e, train_losses[-1], test_losses[-1], train_accs[-1], test_accs[-1])) 
    
print('epoch: {:4d} - train_loss: {:8.4f} - test_loss {:8.4f} - train_acc: {:.4f} - test_cc {:.4f} '.format(e, train_losses[-1], test_losses[-1], train_accs[-1], test_accs[-1])) 

# convert lists to tensors
train_losses = torch.tensor(train_losses)
test_losses = torch.tensor(test_losses)
train_accs = torch.tensor(train_accs)
test_accs = torch.tensor(test_accs)

# plot results
plot_results(model, train_losses, test_losses, train_accs, test_accs)
plt.show()
