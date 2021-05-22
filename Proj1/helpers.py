import torch
from torch import nn

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime as datetime
from datetime import datetime


@ torch.no_grad()
def compute_acc(model, data_loader):
    model.eval()  
    # print(model.training)

    nb_errors = 0
    for input_data, target, _ in iter(data_loader):
        _,_,output = model(input_data)
        for i, out in enumerate(output):
            pred_target = out.max(0)[1].item()
            if (target[i]) != pred_target:
                nb_errors += 1

    error_rate = nb_errors/len(data_loader.dataset)
    # print(len(data_loader.dataset))
    return 1 - error_rate

@ torch.no_grad()
def eval_model(model , data_loader):
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')

    losses = []
    for input_data, target, class_data in iter(data_loader):
        output_aux1, output_aux2, output = model(input_data)
        loss_aux1 = criterion(output_aux1, class_data[:,0])
        loss_aux2 = criterion(output_aux2, class_data[:,1])
        # total losses
        # loss = criterion(output, target) + 0.75*loss_aux1 + 0.75*loss_aux2
        lam = 0.5
        loss = lam * criterion(output, target) + loss_aux1 + loss_aux2
        losses.append(loss)
    
    return torch.cat(losses).mean()


@torch.no_grad()
def compute_error_rate(output, target):
    return 1/output.size(0) * (torch.max(output, 1)[1] != target).long().sum()

# @torch.no_grad()
# def compute_acc(output, target):

#     return 1/output.size(0) * (torch.max(output, 1)[1] == target).long().sum()



# def compute_acc(model, data_loader):
#     return 1 - compute_error_rate(model, data_loader)

def get_device():
    """
    get the device in which tensors, modules and criterions will be stored
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device



def get_str_results(epoch=None, train_loss=None, test_loss=None, train_acc=None, test_acc=None):
    to_print=''

    if epoch is not None:
        to_print += 'epoch: {:3d} '.format(epoch)
    
    if train_loss is not None:
        to_print += '- train_loss: {:6.4f} '.format(train_loss)
                        
    if test_loss is not None:
        to_print += '- test_loss: {:6.4f} '.format(test_loss)

    if train_acc is not None:
        to_print += '- train_acc: {:.4f} '.format(train_acc)
    
    if test_acc is not None:
        to_print += '- test_acc: {:.4f} '.format(test_acc)
    
    return to_print

def plot_results(model, hl, h2, do, log2_bs, eta ,train_losses, test_losses, train_accs, test_accs, savefig=False):
    
    # Get the means and std
    train_losses_mean = train_losses.mean(axis=0)
    train_losses_std = train_losses.std(axis=0)
    test_losses_mean = test_losses.mean(axis=0)
    test_losses_std = test_losses.std(axis=0)
    train_accs_mean = train_accs.mean(axis=0)
    train_accs_std = train_accs.std(axis=0)
    test_accs_mean = test_accs.mean(axis=0)
    test_accs_std = test_accs.std(axis=0)

    print('train acc - mean: {:.3f} std: {:.3f} || test acc - mean: {:.3f} std: {:.3f}'.format(train_accs_mean[-1],train_accs_std[-1],test_accs_mean[-1],test_accs_std[-1]))

    fontsize = 10
    fig, axs = plt.subplots(2, 1, sharex=True)
    
    # write the title
    # fig.suptitle(model.get_str_parameters(), fontsize=fontsize+1)
    fig.suptitle("{} \n hl: {}, h2: {}, do: {:.3f}, bs: {}, eta: {:.4E}".format(type(model).__name__, hl, h2, do, 2**log2_bs, eta), fontsize=fontsize+1)

    # plot the train loss
    axs[0].errorbar(torch.arange(0,train_losses_mean.size(0)), train_losses_mean, train_losses_std/2, label='train_loss', capsize=3)
    axs[0].errorbar(torch.arange(0,test_losses_mean.size(0)), test_losses_mean, test_losses_std/2, label='test_loss', capsize=3)

    axs[0].set_ylabel('loss', fontsize=fontsize)
    axs[0].legend()
    axs[0].grid(True)

    # plot the train_error_rate and test_error_rate
    axs[1].errorbar(torch.arange(0,train_accs_mean.size(0)), train_accs_mean ,train_accs_std/2, label='train_acc', capsize=3)
    axs[1].errorbar(torch.arange(0,test_accs_mean.size(0)), test_accs_mean ,test_accs_std/2, label='test_acc', capsize=3)

    axs[1].set_ylabel('accuracy', fontsize=fontsize)
    axs[1].set_xlabel('epochs', fontsize=fontsize)
    axs[1].set_ylim(0.5, 1.1)
    axs[1].legend()
    axs[1].grid(True)
       
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs = plt.gca()
    
    if savefig:
        date = datetime.now()
        plt.savefig('./Proj1/plots/{}_{:02d}_{:02d}_{}_{:02d}_{:02d}.png'.format(type(model).__name__,
                                    date.day,
                                    date.month,
                                    date.year,
                                    date.hour,
                                    date.minute))
