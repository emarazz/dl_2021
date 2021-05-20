from improved_base_netA import *
from helpers import *
# import seaborn as sns
# sns.set()

# hl, h2, do, log2_bs, eta = binary_search_BaseNet(   hidden_layers1 = [512],
#                                                     hidden_layers2 = [512],
#                                                     dropout_probabilities = [0.125],
#                                                     log2_batch_sizes = [4, 7],
#                                                     etas = [0.075] )

hl, h2, do, log2_bs, eta = binary_search_BaseNet(   hidden_layers1 = [10, 512],
                                                    hidden_layers2 = [10, 512],
                                                    dropout_probabilities = [0, 0.5],
                                                    log2_batch_sizes = [4, 7],
                                                    etas = [1e-3, 1e-1] )

# hl, h2, do, log2_bs, eta = binary_search_BaseNet(   hidden_layers1 = [512],
#                                                     hidden_layers2 = [512],
#                                                     dropout_probabilities = [0.125],
#                                                     log2_batch_sizes = [6],
#                                                     etas = [0.075] )


train_losses, train_error_rates, test_error_rates = eval_BaseNet(hl, h2, do, log2_bs, eta)
print(train_losses)

# path = './BaseNet_tensors_to_plot.pt'
# train_losses, train_error_rates, test_error_rates = torch.load(path)

# parameters of plot_results are only for plotting
plot_results(hl=512, h2=512, do=0.125, log2_bs=6, eta=0.075,
            train_losses=train_losses, train_error_rates=train_error_rates, test_error_rates=test_error_rates) 
plt.show()