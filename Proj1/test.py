from models import *
from helpers import *
from train import *
# from plot import *   # plotting
'''
EE-559 Deep Learning - Proj1
    - Erick Maraz 
    - Daniel Filipe Nunes Silva
    - Gloria Dal Santo
'''

'''
======================================================================================================
                                        Model Selection
            . uncomment one of the following blocks of code to use the desired model
            . the value of the hyper-parameters has been found through binary search
            . available models: 
                - BaseNetMLP: MultiLayer Perceptron 
                - BaseNetCNN: Convolutional Neural Network + Feed Forward Neural Network
                - AuxNet: CNN + FFNN with auxiliary losses
                - SiameseNet: CNN + FFNN with auxiliary losses and weight sharing   
======================================================================================================
'''
# # -------- BaseNet MLP -------- #
# cls = BaseNetMLP
# # model parameters
# h1, h2, h3, do = 512, 512, 'NaN', 0
# # train parameters 
# log2_bs, eta, optim = 6, 0.1, 'SGD'

# # ----- BaseNet CNN + FFNN ----- #
# cls = BaseNetCNN
# # model parameters
# h1, h2, h3, do = 64, 64, 'NaN', 0
# # train parameters 
# log2_bs, eta, optim = 6, 0.1, 'SGD'

# # ----------- AuxNet ----------- #
# cls = AuxNet
# # model parameters
# h1, h2, h3, do = 64, 288, 64, 0
# # train parameters 
# log2_bs, eta, optim = 6, 0.001, 'Adam'
# alpha, beta = 0.1, 1

# # --------- SiameseNet --------- #
cls = SiameseNet
# model parameters
h1, h2, h3, do = 64, 120, 120, 0
# train parameters 
log2_bs, eta, optim = 6, 0.0025, 'Adam'
alpha, beta = 0.1, 1

epochs = 30
'''
======================================================================================================
                                        Training and Testing
                    . cronstruct the desired model
                    . load data from MNIST and create train and test data loaders
                    . train the model and get the performance estimators 
                    note:   number of evaluation runs is set to 1 by default 
                            consider changing 'evaluation_rounds'
======================================================================================================
'''
# set the number of evaluation rounds
evaluation_rounds = 1
# construct the model
model = cls(nb_hidden1 = h1,
            nb_hidden2 = h2,
            nb_hidden3 = h3,
            dropout_prob=do)

# load data and construct the data loaders 
train_loader, test_loader = get_data(N=1000, batch_size=2**log2_bs, shuffle=True)

# run the network and get permformance estimatiors on 15 runs 
train_losses, test_losses, train_accuracy, test_accuracy = run_Net(cls, h1, h2, h3, do, log2_bs, eta, epochs, 
                                                                            optim=optim, save_tensors=True, evaluation_rounds=evaluation_rounds)
# compute mean and standard deviation
train_losses_mean = train_losses.mean(axis=0)
train_losses_std = train_losses.std(axis=0)
test_losses_mean = test_losses.mean(axis=0)
test_losses_std = test_losses.std(axis=0)
train_accs_mean = train_accuracy.mean(axis=0)
train_accs_std = train_accuracy.std(axis=0)
test_accs_mean = test_accuracy.mean(axis=0)
test_accs_std = test_accuracy.std(axis=0)

print('='*70)
print('PERFORMANCE ESTIMATION - {} rounds'.format(evaluation_rounds))
print('-'*70)
print('mean_train_loss: {:.4f}, mean_test_loss: {:.4f} || mean_train_acc: {:.4f}, mean_test_acc: {:.4f}'.format(train_losses_mean[-1], test_losses_mean[-1], train_accs_mean[-1], test_accs_mean[-1]))
print('std_train_loss: {:.4f}, std_test_loss: {:.4f} || std_train_acc: {:.4f}, std_test_acc: {:.4f}'.format(train_losses_std[-1], test_losses_std[-1], train_accs_std[-1], test_accs_std[-1]))
print('='*70)
print('\n')


'''
======================================================================================================
                                        Extra Code                                               
======================================================================================================
'''                            
'''Plotting
If you want to plot the test/train accuracy and loss load plot_results from plot.py and uncomment the following lines
'''                                                                            
# plot the the train/test accuracy and loss
# plot_results(model, h1, h2, h3, do, log2_bs, eta,
#             train_losses=train_losses, test_losses=test_losses, train_accs=train_accuracy, test_accs=test_accuracy, savefig=True) 
# plt.show()

'''Binary Search
uncomment the following code to run a binary search on the hyper-parameters
for more details on binary_search_Net() go to 'train.py'
'''
# hl, h2, h3, do, log2_bs, eta = binary_search_Net(cls,   nb_hidden1 = [64, 512],
#                                                         nb_hidden2 = [64, 512],
#                                                         nb_hidden2 = [64, 512],
#                                                         log2_batch_sizes = [6, 7],
#                                                         etas = [0.001, 0.1],
#                                                         epochs = epochs, optim='Adam')
