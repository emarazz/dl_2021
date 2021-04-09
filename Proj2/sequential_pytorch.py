import torch
import torch.nn as nn

import dlc_practical_prologue as prologue



train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels = True,
                                                                        normalize = True)
nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)

zeta = 0.90

train_target = train_target * zeta
test_target = test_target * zeta

model = nn.Sequential(
    nn.Linear(784, 100), nn.Tanh(),
    nn.Linear(100, 100), nn.Tanh(),
    nn.Linear(100, 100), nn.Tanh(),
    nn.Linear(100,10), nn.ReLU()
)
criterion = nn.MSELoss(reduction='sum')

nb_epochs = 1000
batch_size = 1000
nb_hidden = 50
eta = 1e-1 / nb_train_samples
epsilon = 1e-6

torch.manual_seed(0)

# model[0].weight.data = torch.empty(nb_hidden, train_input.size(1)).normal_(0, epsilon)
# model[0].bias.data = torch.empty(nb_hidden).normal_(0, epsilon)
# model[2].weight.data = torch.empty(nb_classes, nb_hidden).normal_(0, epsilon)
# model[2].bias.data = torch.empty(nb_classes).normal_(0, epsilon)

# print(model)
# for param in model.parameters():
#     print(param.data)

# output = model(train_input)
# loss = criterion(output, train_target)
# print(output)
# print(loss)

# model.zero_grad()
# loss.backward()
# print(model[0].weight.grad)

# print('{:.16f}'.format(output.sum()))
# print('{:.10f}'.format(criterion(output, train_target)))

for e in range(nb_epochs):
    output = model(train_input)
    loss = criterion(output, train_target)

    with torch.no_grad():
        test_prediction = model(test_input)
        test_loss = criterion(test_prediction, test_target)

    print('epoch: {:4d} train_loss: {:8.4f} test_loss: {:8.4f}'.format(e+1, loss, test_loss))

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= eta * p.grad

