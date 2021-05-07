import torch
import torch.utils.data

def train_BaseNet(model, epochs, eta, train_loader, test_loader, eval_mode=False):
    losses = []
    train_error_rates = []
    test_error_rates = []

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
        
    for _ in range(epochs):
        for input_data, target, _ in iter(train_loader):
            output = model(input_data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss)
        train_error_rates.append(compute_error_rate(model, train_loader))
        test_error_rates.append(compute_error_rate(model, test_loader))

    return losses, train_error_rates, test_error_rates

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
