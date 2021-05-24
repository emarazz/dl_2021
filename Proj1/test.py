from models import *
from helpers import *
from train import *

'''
# ========================================================================================
# # BaseNetCNN
# ========================================================================================
'''
# cls = BaseNetCNN
# epochs = 30
# # model parameters
# h1, h2, h3, do = 64, 64, 'NaN', 0
# # train parameters 
# log2_bs, eta, optim = 6, 0.1, 'SGD'

# # BINARY SEARCH
# # hl, h2, h3, do, log2_bs, eta = binary_search_Net(cls,    nb_hidden1 = [64, 512],
# #                                                         nb_hidden2 = [64, 512],
# #                                                         log2_batch_sizes = [6, 7],
# #                                                         etas = [0.001, 0.1],
# #                                                         epochs = epochs, optim='SGD')

# model = cls(    nb_hidden1=h1,
#                 nb_hidden2=h2,
#                 dropout_prob=do)

# train_loader, test_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True)

# train_losses, test_losses, train_error_rates, test_error_rates = run_Net(cls, h1, h2, h3, do, log2_bs, eta, epochs, 
#                                                                             optim=optim, save_tensors=True)

# plot_results(model, h1, h2, h3, do, log2_bs, eta,
#             train_losses=train_losses, test_losses=test_losses, train_accs=train_error_rates, test_accs=test_error_rates, savefig=True) 
# plt.show()
'''
# ========================================================================================
# # BaseNetMLP
# ========================================================================================
'''
# cls = BaseNetMLP
# epochs = 30
# # model parameters
# h1, h2, h3, do = 512, 512, 'NaN', 0
# # train parameters 
# log2_bs, eta, optim = 6, 0.1, 'SGD'

# # BINARY SEARCH
# # hl, h2, h3, do, log2_bs, eta = binary_search_Net(cls,    nb_hidden1 = [64, 512],
# #                                                         nb_hidden2 = [64, 512],
# #                                                         log2_batch_sizes = [6, 7],
# #                                                         etas = [0.001, 0.1],
# #                                                         epochs = epochs, optim='SGD')

# model = cls(    nb_hidden1=h1,
#                 nb_hidden2=h2,
#                 dropout_prob=do)

# train_loader, test_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True)

# train_losses, test_losses, train_error_rates, test_error_rates = run_Net(cls, h1, h2, h3, do, log2_bs, eta, epochs, 
#                                                                             optim=optim, save_tensors=True)

# plot_results(model, h1, h2, h3, do, log2_bs, eta,
#             train_losses=train_losses, test_losses=test_losses, train_accs=train_error_rates, test_accs=test_error_rates, savefig=True) 
# plt.show()
'''
# ========================================================================================
# # AuxNet
# ========================================================================================
'''
# cls = AuxNet
# epochs = 30
# # model parameters
# h1, h2, h3, do = 64, 288, 64, 0
# # train parameters 
# log2_bs, eta, optim = 6, 0.001, 'Adam'
# alpha, beta = 0.1, 1

# # BINARY SEARCH
# # hl, h2, h3, do, log2_bs, eta = binary_search_Net(cls,    nb_hidden1 = [64, 512],
# #                                                         nb_hidden2 = [64, 512],
# #                                                         log2_batch_sizes = [6, 7],
# #                                                         etas = [0.001, 0.1],
# #                                                         epochs = epochs, optim='SGD',
# #                                                         alpha = alpha, beta = beta)

# model = cls(    nb_hidden1=h1,
#                 nb_hidden2=h2,
#                 dropout_prob=do)

# train_loader, test_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True)

# train_losses, test_losses, train_error_rates, test_error_rates = run_Net(cls, h1, h2, h3, do, log2_bs, eta, epochs, 
#                                                                             optim=optim, alpha = alpha, beta = beta, save_tensors=True)

# plot_results(model, h1, h2, h3, do, log2_bs, eta,
#             train_losses=train_losses, test_losses=test_losses, train_accs=train_error_rates, test_accs=test_error_rates, savefig=True) 
# plt.show()
'''
# ========================================================================================
# # SiameseNet
# ========================================================================================
'''
cls = SiameseNet
epochs = 30
# model parameters
h1, h2, h3, do = 64, 120, 120, 0
# train parameters 
log2_bs, eta, optim = 6, 0.0025, 'Adam'
alpha, beta = 0.1, 1

# BINSRY SEARCH
# hl, h2, h3, do, log2_bs, eta = binary_search_Net(cls,    nb_hidden1 = [64, 512],
#                                                         nb_hidden2 = [64, 512],
#                                                         log2_batch_sizes = [6, 7],
#                                                         etas = [0.001, 0.1],
#                                                         epochs = epochs, optim='SGD',
#                                                         alpha = alpha, beta = beta)

model = cls(    nb_hidden1=h1,
                nb_hidden2=h2,
                dropout_prob=do)

train_loader, test_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True)

train_losses, test_losses, train_error_rates, test_error_rates = run_Net(cls, h1, h2, h3, do, log2_bs, eta, epochs, 
                                                                            optim=optim, alpha = alpha, beta = beta, save_tensors=True)

plot_results(model, h1, h2, h3, do, log2_bs, eta,
            train_losses=train_losses, test_losses=test_losses, train_accs=train_error_rates, test_accs=test_error_rates, savefig=True) 
plt.show()
