import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime as datetime


@ torch.no_grad()
def compute_error_rate(model, data_loader):
    model.eval()  
    nb_errors = 0

    for input_data, target, _ in iter(data_loader):
        output = model(input_data)
        for i, out in enumerate(output):
            pred_target = out.max(0)[1].item()
            if (target[i]) != pred_target:
                nb_errors += 1

    error_rate = nb_errors/len(data_loader.dataset)

    return error_rate

def compute_acc(model, data_loader):
    return 1 - compute_error_rate(model, data_loader)

def get_device():
    """
    get the device in which tensors, modules and criterions will be stored
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def get_str_results(e, train_loss, train_error_rate, test_error_rate):
    return 'epoch: {:3d} - train_loss: {:6.4f} - train_error_rate: {:5.4f} - test_error_rate: {:5.4f}'.format(
                        e, train_loss, train_error_rate, test_error_rate)

def plot_results(model, path, savefig=False):
    # Load the tensors
    train_losses, train_error_rates, test_error_rates = torch.load(path)

    # Get the means and std
    train_losses_mean = train_losses.mean(axis=0)
    train_losses_std = train_losses.std(axis=0)
    train_error_rates_mean = train_error_rates.mean(axis=0)
    train_error_rates_std = train_error_rates.std(axis=0)
    test_error_rates_mean = test_error_rates.mean(axis=0)
    test_error_rates_std = test_error_rates.std(axis=0)

    print('train error rate - mean: {:.3f} std: {:.3f} || test error rate - mean: {:.3f} std: {:.3f}'.format(train_error_rates_mean[-1],train_error_rates_std[-1],test_error_rates_mean[-1],test_error_rates_std[-1]))

    fontsize = 10
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.suptitle(model.get_str_parameters(), fontsize=fontsize+1)

    # plot the train loss
    axs[0].errorbar(torch.arange(0,train_losses_mean.size(0)), train_losses_mean ,train_losses_std/2, label='train_loss', capsize=3)
    
    axs[0].set_ylabel('loss', fontsize=fontsize)
    axs[0].legend()
    axs[0].grid(True)

    # plot the train_error_rate and test_error_rate
    axs[1].errorbar(torch.arange(0,train_error_rates_mean.size(0)), train_error_rates_mean ,train_error_rates_std/2, label='train_error_rate', capsize=3)
    axs[1].errorbar(torch.arange(0,test_error_rates_mean.size(0)), test_error_rates_mean ,test_error_rates_std/2, label='test_error_rate', capsize=3)

    axs[1].set_ylabel('error rate', fontsize=fontsize)
    axs[1].set_xlabel('epochs', fontsize=fontsize)
    axs[1].set_ylim(0, 0.6)
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
