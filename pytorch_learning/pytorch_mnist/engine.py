import torch
from torch import nn
import copy

def train_one_model(z, y_true, loss_fn, optimizer):
    loss = loss_fn(z, y_true)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, val_dataloader, best_accuracy, lost_fn):
    n_acc = 0
    n_all = 0
    best_model_state = None
    device = next(model.parameters()).device
    loss_mean = 0.0
    n_iter = 0
    for x, y in val_dataloader:
        x, y = x.to(device), y.to(device)  # Đưa data lên GPU
        z = model(x)
        z = torch.argmax(z, dim=1)

        n_acc = n_acc + (z == y).sum().item()
        n_all = n_all + y.size(0)

        loss_mean += lost_fn(z, y)
        n += 1
        
    current_acc = n_acc / n_all
    print(f'   validate accuracy : {current_acc}')

    loss_mean /= n
    
    if best_accuracy < current_acc:
        best_accuracy = current_acc
        best_model_state = copy.deepcopy(model.state_dict())
        print(f'-> preserved model with acc: {best_accuracy}')
        
    return best_model_state, current_acc, loss_mean

def test_phase(model, test_dataloader, device):
    t = 0
    all_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)  # Đưa data lên GPU
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            t = t + (y_pred == y).sum().item()
            all_samples = all_samples + y.size(0)
    return t / all_samples