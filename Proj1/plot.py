import torch
from torch import nn

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime as datetime
from datetime import datetime


def plot_results(model, hl, h2, h3, do, log2_bs, eta ,train_losses, test_losses, train_accs, test_accs, savefig=False):
    '''plot the test and train accuracy and loss over the epochs
    '''
    # Get the means and std
    train_losses_mean = train_losses.mean(axis=0)
    train_losses_std = train_losses.std(axis=0)
    test_losses_mean = test_losses.mean(axis=0)
    test_losses_std = test_losses.std(axis=0)
    train_accs_mean = train_accs.mean(axis=0)
    train_accs_std = train_accs.std(axis=0)
    test_accs_mean = test_accs.mean(axis=0)
    test_accs_std = test_accs.std(axis=0)

    # print('train acc - mean: {:.4f} std: {:.4f} || test acc - mean: {:.4f} std: {:.4f}'.format(train_accs_mean[-1],train_accs_std[-1],test_accs_mean[-1],test_accs_std[-1]))

    fontsize = 10
    fig, axs = plt.subplots(2, 1, sharex=True)
    
    # write the title
    # fig.suptitle(model.get_str_parameters(), fontsize=fontsize+1)
    if type(model).__name__ == 'BaseNetCNN' or type(model).__name__ == 'BaseNetMLP':
        fig.suptitle("{} \n hl: {}, h2: {}, do: {:.3f}, bs: {}, eta: {:.4E}".format(type(model).__name__, hl, h2, do, 2**log2_bs, eta), fontsize=fontsize+1)
    elif type(model).__name__ == 'AuxNet' or type(model).__name__ == 'SiameseNet': 
        fig.suptitle("{} \n hl: {}, h2: {}, h3: {}, do: {:.3f}, bs: {}, eta: {:.4E}".format(type(model).__name__, hl, h2, h3, do, 2**log2_bs, eta), fontsize=fontsize+1)
    else:
        print('[error] model not recognized')
        return

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
        plt.savefig('./plots/{}_{:02d}_{:02d}_{}_{:02d}_{:02d}.png'.format(type(model).__name__,
                                    date.day,
                                    date.month,
                                    date.year,
                                    date.hour,
                                    date.minute))