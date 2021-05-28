import torch
from torch import nn

@ torch.no_grad()
def compute_acc(model, data_loader):
    '''compute accuracy
    feed 'model' with the input data from 'data_loader'
    compute the accuracy of the model's output
    '''
    model.eval()  
    nb_errors = 0
    for input_data, target, _ in iter(data_loader):
        
        if model.get_aux_info() == True:
            _,_,output = model(input_data)
        else: 
            output = model(input_data)
   
        for i, out in enumerate(output):
            pred_target = out.max(0)[1].item()
            if (target[i]) != pred_target:
                nb_errors += 1

    error_rate = nb_errors/len(data_loader.dataset)
    return 1 - error_rate

@ torch.no_grad()
def eval_model(model, data_loader, alpha = 1, beta = 1):
    '''evaluate model
    train 'model' with one epoch and compute the losses 
    '''
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')

    losses = []
    if model.get_aux_info() == False:
        for input_data, target, _ in iter(data_loader):
            output = model(input_data)
            loss = criterion(output, target)
            losses.append(loss)       
    else: 
        for input_data, target, class_data in iter(data_loader):
            output_aux1, output_aux2, output = model(input_data)
            loss_aux1 = criterion(output_aux1, class_data[:,0])
            loss_aux2 = criterion(output_aux2, class_data[:,1])

            loss = alpha * criterion(output, target) + beta * ( loss_aux1 + loss_aux2 )

            losses.append(loss)
    
    return torch.cat(losses).mean()


@torch.no_grad()
def compute_error_rate(output, target):
    ''' compare 'output' with 'target' and compute the error rate
    '''
    return 1/output.size(0) * (torch.max(output, 1)[1] != target).long().sum()

def get_device():
    '''get the device in which tensors, modules and criterions will be stored
    '''
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def get_str_results(epoch=None, train_loss=None, test_loss=None, train_acc=None, test_acc=None):
    '''construct the string that has to be print 
    '''
    to_print=''

    if epoch is not None:
        to_print += 'epoch: {:3d} '.format(epoch)
    
    if train_loss is not None:
        to_print += '- train_loss: {:6.4f} '.format(train_loss)
                        
    if test_loss is not None:
        to_print += '- test_loss: {:6.4f} '.format(test_loss)

    if train_acc is not None:
        to_print += '- train_acc: {:.4f} '.format(train_acc)
    
    if test_acc is not None:
        to_print += '- test_acc: {:.4f} '.format(test_acc)
    
    return to_print


def count_parameters(model):
    '''count the total number of parameters in 'model'
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




