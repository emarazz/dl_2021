from improved_base_netA import *
from improved_aux_net import *
from improved_siamese_net import *

### Example for binary search

# hl, h2, h3, do, log2_bs, eta = binary_search_AuxNet(    hidden_layers1 = [512],
#                                                         hidden_layers2 = [512],
#                                                         hidden_layers3 = [512],
#                                                         dropout_probabilities = [0.125],
#                                                         log2_batch_sizes = [6],
#                                                         etas = [0.075],
#                                                         epochs = epochs )

# print('Several runs...')                       
# train_losses, test_losses, train_error_rates, test_error_rates = run_AuxNet(hl, h2, h3, do, log2_bs, eta, epochs)

# plot_results(model, hl=hl, h2=h2, do=0.1, log2_bs=log2_batch_size, eta=eta,
#             train_losses=train_losses, test_losses=test_losses, train_error_rates=train_error_rates, test_error_rates=test_error_rates) 
# plt.show()
####
####

# # ========================================================================================
# # # BaseseNet
# # ========================================================================================
# model = AuxNet( nb_hidden1=128,
#                     nb_hidden2=512,
#                     dropout_prob=0  )

# log2_batch_size = 7
# train_loader, test_loader = get_data(N=1000, batch_size=2**log2_batch_size, shuffle=True)

# epochs = 50
# eta = 0.003
# train_losses, test_losses, train_error_rates, test_error_rates = run_AuxNet(hl=128, h2=512, do=0, log2_bs=log2_batch_size, eta=eta, epochs=epochs)

# plot_results(model, hl=128, h2=512, do=512, log2_bs=log2_batch_size, eta=eta,
#             train_losses=train_losses, test_losses=test_losses, train_error_rates=train_error_rates, test_error_rates=test_error_rates) 
# plt.show()

# # ========================================================================================
# # # AuxseNet
# # ========================================================================================
# model = AuxNet( nb_hidden1=128,
#                     nb_hidden2=512,
#                     nb_hidden3=512,
#                     dropout_prob=0  )

# log2_batch_size = 7
# train_loader, test_loader = get_data(N=1000, batch_size=2**log2_batch_size, shuffle=True)

# epochs = 50
# eta = 0.003
# train_losses, test_losses, train_error_rates, test_error_rates = run_AuxNet(hl=128, h2=512, h3=512, do=0, log2_bs=log2_batch_size, eta=eta, epochs=epochs)

# plot_results(model, hl=128, h2=512, do=512, log2_bs=log2_batch_size, eta=eta,
#             train_losses=train_losses, test_losses=test_losses, train_error_rates=train_error_rates, test_error_rates=test_error_rates) 
# plt.show()

# ========================================================================================
# # # SiameseNet
# ========================================================================================
model = SiameseNet( nb_hidden1=128,
                    nb_hidden2=512,
                    nb_hidden3=512,
                    dropout_prob=0  )

log2_batch_size = 7
train_loader, test_loader = get_data(N=1000, batch_size=2**log2_batch_size, shuffle=True)

epochs = 50
eta = 0.003
train_losses, test_losses, train_error_rates, test_error_rates = run_SiameseNet(hl=128, h2=512, h3=512, do=0, log2_bs=log2_batch_size, eta=eta, epochs=epochs)

plot_results(model, hl=128, h2=512, do=0, log2_bs=log2_batch_size, eta=eta,
            train_losses=train_losses, test_losses=test_losses, train_error_rates=train_error_rates, test_error_rates=test_error_rates) 
plt.show()

# ========================================================================================
# # # CNN_AUX_WS
# ========================================================================================

# model = CNN_AUX_WS( nb_hidden1=0,
#                     nb_hidden2=0,
#                     nb_hidden3=0,
#                     dropout_prob=0  )


# train_loader, test_loader = get_data(N=1000, batch_size=100, shuffle=True)
# train_losses, train_error_rates = train_SiameseNet(model, 0.003, 50,train_loader)
# test_losses, test_error_rates = eval_SiameseNet(model, 50,test_loader)
# plt.plot(train_losses)
# plt.plot(test_losses)
# plt.plot(train_error_rates)
# plt.plot(test_error_rates)
# plt.show()



# log2_batch_size = 7
# train_loader, test_loader = get_data(N=1000, batch_size=2**log2_batch_size, shuffle=True)

# epochs = 50
# eta = 0.003
# train_losses, test_losses, train_error_rates, test_error_rates = run_SiameseNet(cls=CNN_AUX_WS,
#                                                                                 log2_bs=log2_batch_size, eta=eta, epochs=epochs,
#                                                                                 hl=0, h2=0, h3=0, do=0)

# plot_results(model, hl=0, h2=0, do=0, log2_bs=log2_batch_size, eta=eta,
#             train_losses=train_losses, test_losses=test_losses, train_error_rates=train_error_rates, test_error_rates=test_error_rates) 
# plt.show()
