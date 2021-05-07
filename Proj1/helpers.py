import torch

def compute_error_rate(model, data_loader):
    nb_errors = 0

    for input_data, target, _ in iter(data_loader):
        output = model(input_data)
        for i, out in enumerate(output):
            pred_target = out.max(0)[1].item()
            if (target[i]) != pred_target:
                nb_errors += 1

    error_rate = nb_errors/len(data_loader.dataset)

    return error_rate

def get_device():
    """
    get the device in which tensors, modules and criterions will be stored
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device
