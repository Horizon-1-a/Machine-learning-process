import torch
from torch import nn
import copy

def train_one_model(z, y_true, loss_fn, optimizer):
    optimizer.zero_grad()
    loss = loss_fn(z, y_true)
    loss.backward()
    optimizer.step()
def evaluate(model, val_dataloader, best_accuracy):
    n_acc = 0
    n_all = 0
    best_model = None
    for x, y in val_dataloader:
        z = model(x)
        z = torch.argmax(z, dim= 1)
        n_acc = n_acc + (z == y).sum().item()
        n_all = n_all + y.size(0)
    if (best_accuracy < n_acc / n_all):
        best_accuracy = n_acc / n_all
        best_model = copy.deepcopy(model.state_dict())
    return best_model
def test_phase(model, test_dataloader):
    t = 0
    all = 0
    for x, y in test_dataloader:
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim= 1)
        t = t + (y_pred == y).sum().item()
        all = all + y.size(0)
    return t / all

