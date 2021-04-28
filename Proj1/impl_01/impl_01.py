import torch
from torch import nn 
from torch.nn import functional as F
import dlc_practical_prologue as prologue

# load data
N = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
# train_input = train_input.narrow(0,0,10).view(10,2,196,-1).view(10,392,-1).view(10,-1)
# print(train_input.size())
# b = 20
# data = train_input.narrow(0,0,b).view(b,392,-1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(392, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.drop = nn.Dropout(0.2)


    def forward(self, x):
        x = F.relu(self.drop(self.fc1(x)))
        x = F.relu(self.drop(self.fc2(x)))
        x = torch.sigmoid(self.drop(self.fc3(x)))

        return x


def train_model(model, train_input, train_target, mini_batch_size, nb_epochs):
    eta = 1.2e-1
    optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size).view(mini_batch_size, -1))
            # output = model(train_input.narrow(0,b,mini_batch_size).view(mini_batch_size,2,196,-1).view(mini_batch_size,392,-1).view(mini_batch_size,-1))
            #output = model(train_input.narrow(0, b, mini_batch_size).view(mini_batch_size,392,-1))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
        
#        print(e, acc_loss)

def compute_nb_errors(model, input, target, target_ohl, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size).view(mini_batch_size,-1))
        #output = model(input.narrow(0,b,mini_batch_size).view(mini_batch_size,2,196,-1).view(mini_batch_size,392,-1).view(mini_batch_size,-1))
        _, predicted_lower = output.max(1)
        for k in range(mini_batch_size):
            if target_ohl[b + k, predicted_lower[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors


 # change device 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


criterion = nn.CrossEntropyLoss()
criterion.to(device)
train_input = train_input.to(device)
train_target = train_target.to(device)
train_classes = train_classes.to(device)
test_input = test_input.to(device)
test_target = test_target.to(device)
test_classes = test_classes.to(device)

train_target_ohl = prologue.convert_to_one_hot_labels(train_target, train_target)
test_target_ohl = prologue.convert_to_one_hot_labels(train_target, test_target)
# tmp = train_input.new_zeros(train_target.size(0),2)
# tmp.scatter_(1, train_target.view(-1,1), 1.0)
# print(tmp[0:10],train_target[0:10])

# modify target into 
# data normalization
mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)
mu, std = test_input.mean(), test_input.std()
test_input.sub_(mu).div_(std)

mini_batch_size = 50
nb_epochs = 50

for k in range(10):
    model = Net()
    model.to(device)
    train_model(model, train_input, train_target, mini_batch_size, nb_epochs)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, test_target_ohl, mini_batch_size)
    nb_train_errors = compute_nb_errors(model, train_input, train_target, train_target_ohl, mini_batch_size)

    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    print('train error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                      nb_train_errors, train_input.size(0)))    
    del model
