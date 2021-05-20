from improved_base_netA import *

epochs = 30

# model = BaseNet(nb_hidden1=512, nb_hidden2=512, dropout_prob=0.125)

# log2_batch_size = 6
# train_loader, test_loader = get_data(N=1000, batch_size=2**log2_batch_size, shuffle=True)

# train_BaseNet(model=model ,eta=0.075, epochs=epochs, train_loader=train_loader, print_results=True)
# eval_BaseNet(model=model, epochs=epochs, test_loader=test_loader, print_results=True)

print('Running binary search...')
print('='*70)

hl, h2, do, log2_bs, eta = binary_search_BaseNet(   hidden_layers1 = [10, 512],
                                                    hidden_layers2 = [10, 512],
                                                    dropout_probabilities = [0, 0.5],
                                                    log2_batch_sizes = [4, 7],
                                                    etas = [1e-3, 1e-1],
                                                    epochs = epochs )

# hl, h2, do, log2_bs, eta = binary_search_BaseNet(   hidden_layers1 = [512],
#                                                     hidden_layers2 = [512],
#                                                     dropout_probabilities = [0.125],
#                                                     log2_batch_sizes = [6],
#                                                     etas = [0.075],
#                                                     epochs = epochs )

# print('Several runs...')                       

train_losses, test_losses, train_error_rates, test_error_rates = run_BaseNet(hl, h2, do, log2_bs, eta, epochs)


# path = './BaseNet_tensors_to_plot.pt'
# train_losses, test_losses, train_error_rates, test_error_rates = torch.load(path)
# print(train_losses)

# parameters of plot_results are only for plotting
plot_results(hl=512, h2=512, do=0.125, log2_bs=6, eta=0.075,
            train_losses=train_losses, test_losses=test_losses, train_error_rates=train_error_rates, test_error_rates=test_error_rates) 
plt.show()