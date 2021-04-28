import torch
import torch.utils.data

"""
ToDo: in optim.SGD add momentum and nesterov options 

"""
def train_BaseNet(model, train_loader ,criterion, eta, epochs, optimizer):
    acc_loss = 0
    loss = []
    if optimizer == "SGD":    
        optimizer = torch.optim.SGD(model.parameters(), lr = eta) 
        
    for e in range(epochs):
        for input_data, target, _ in iter(train_loader):
            output = model(input_data)
            temp = criterion(output, target)
            loss.append(temp)
            acc_loss = acc_loss + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return acc_loss, loss 

def compute_acc_BaseNet(model, data_loader):
    nb_errors = 0

    for input_data, target, _ in iter(data_loader):
        output = model(input_data)
        for i, out in enumerate(output):
            pred_target = out.max(0)[1].item()
            if (target[i]) != pred_target:
                nb_errors += 1

    return 1 - nb_errors / len(data_loader.dataset)
