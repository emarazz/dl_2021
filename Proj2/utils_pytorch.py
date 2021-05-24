import torch

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Set autograd off
# torch.set_grad_enabled(False) # comment for test2_pytorch.py
import torch.nn as nn # uncomment for test2_pytorch.py

def is_inside(input, cols=2):
    """ Returns 0 if the input is outside a disk centered at (0.5, 0.5) of radius 1 / sqrt(2*pi)) and 1 inside.
    """

    pi = torch.acos(torch.tensor(-1.)) # Calculate pi :p
    center = torch.tensor([0.5, 0.5])
    radius = 1/(torch.sqrt(2 * pi))

    output = torch.zeros(input.size(0), cols, dtype=torch.long)
    mask = torch.norm(input.view(-1,2) - center, dim=1) < radius
    output[mask] = 1
    one_hot_labels = torch.eye(cols)[output[:,0]] # One hot encoding

    return one_hot_labels

def is_inside2(input):
    """ Returns 0 if the input is outside a disk centered at (0.5, 0.5) of radius 1 / sqrt(2*pi)) and 1 inside.
    """

    pi = torch.acos(torch.tensor(-1.)) # Calculate pi :p
    center = torch.tensor([0.5, 0.5])
    radius = 1/(torch.sqrt(2 * pi))

    output = torch.zeros(input.size(0), dtype=torch.long)
    mask = torch.norm(input.view(-1,2) - center, dim=1) < radius
    output[mask] = 1
    
    return output.view(input.size(0),-1)

@torch.no_grad()
def eval_model_pytorch(model, input_data, input_target):
    """ Evaluates the model and returns the loss.
    """

    model.eval()
    # print(model.training)
    criterion = nn.MSELoss()

    output = model(input_data)
    loss = criterion(output, input_target)
    
    return loss

@torch.no_grad()
def compute_acc_pytorch(model, input_data, input_target):
    """ Computes the accuracy of the model.
    """

    model.eval()  
    # print(model.training)

    nb_errors = 0
          
    output = model(input_data)
    for i, out in enumerate(output):
        pred_target = (out > 0.5).long()
        if (input_target[i]) != pred_target:
            nb_errors += 1

    error_rate = nb_errors/input_target.size(0)
    
    return 1 - error_rate

def plot_results(model, train_losses, test_losses, train_accs, test_accs, title='MLP - 3 layers of 25 units', savefig=False):
    
    # # Get the means and std
    # train_losses_mean = train_losses.mean(axis=0)
    # train_losses_std = train_losses.std(axis=0)
    # test_losses_mean = test_losses.mean(axis=0)
    # test_losses_std = test_losses.std(axis=0)
    # train_accs_mean = train_accs.mean(axis=0)
    # train_accs_std = train_accs.std(axis=0)
    # test_accs_mean = test_accs.mean(axis=0)
    # test_accs_std = test_accs.std(axis=0)

    fontsize = 10
    fig, axs = plt.subplots(2, 1, sharex=True)
    
    # write the title
    fig.suptitle(title)
    # plot the train loss
    axs[0].plot(train_losses, label='train_loss')
    axs[0].plot(test_losses, label='test_loss')

    axs[0].set_ylabel('loss', fontsize=fontsize)
    axs[0].legend()
    axs[0].grid(True)

    # plot the train_error_rate and test_error_rate
    axs[1].plot(train_accs, label='train_acc')
    axs[1].plot(test_accs, label='test_acc')

    axs[1].set_ylabel('accuracy', fontsize=fontsize)
    axs[1].set_xlabel('epochs', fontsize=fontsize)
    axs[1].set_ylim(0.5, 1.1)
    axs[1].legend()
    axs[1].grid(True)
       
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs = plt.gca()
    
    if savefig:
        date = datetime.now()
        plt.savefig('./plots/{}_{:02d}_{:02d}_{}_{:02d}_{:02d}.png'.format(type(model).__name__,
                                    date.day,
                                    date.month,
                                    date.year,
                                    date.hour,
                                    date.minute))
