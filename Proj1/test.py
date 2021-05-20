from improved_base_netA import *
from helpers import *
# import seaborn as sns
# sns.set()


# eval_BaseNet(   hidden_layers1=[512],
#                 hidden_layers2=[512],
#                 log2_batch_sizes=[6],
#                 etas=[0.075],
#                 dropout_probabilities=[0.125]    
#                 )

model = BaseNet(512,512,0.125)

plot_results(model, './BaseNet_tensors_to_plot.pt')
plt.show()