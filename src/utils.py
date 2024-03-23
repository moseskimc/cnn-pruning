import torch


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)  # get the arg max indices per row
    return torch.tensor(
        torch.sum(preds == labels).item() / len(preds)
    )  # take average over matches


def get_default_device():
    """Return GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [
            to_device(x, device) for x in data
        ]  # use recursion to move data from all dimensions
    return data.to(device, non_blocking=True)


@torch.no_grad()
def evaluate_model(model, test_loader):
    model.eval()
    outputs = [
        model.validation_step(batch) for batch in test_loader
    ]  # compute val loss/acc per batch
    return model.validation_epoch_end(outputs)  # return average loss/acc


def train_model(epochs,
                lr,
                model,
                train_loader,
                test_loader,
                opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:  # train on batch
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate_model(model, test_loader)  # validate
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)  # print metrics for each epoch
        history.append(result)
    return history
