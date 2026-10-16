import torch
from torch import nn
from data import get
from model import Mymodel
from engine import train_one_model, evaluate, test_phase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get()
Train_model = Mymodel(256).to(device)
best_model = Mymodel(256).to(device)
optimizer = torch.optim.SGD(Train_model.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()
epochs = 30

best_acc = 0.0

# train 
for i in range (epochs):
    Train_model.train()
    print (f'epoch {i}: ')
    for x, y_true in train_loader:
        optimizer.zero_grad()
        x, y_true = x.to(device), y_true.to(device)
        z = Train_model(x)
        train_one_model(z, y_true= y_true, loss_fn= loss_fn, optimizer= optimizer)
    #validating 
    with torch.no_grad():
        Train_model.eval()
        res, cur_acc = evaluate(val_dataloader= val_loader, model= Train_model, best_accuracy= best_acc)
        if (res != None):
            best_model.load_state_dict(res)
            best_model.train()
            best_acc = cur_acc

real_acc = test_phase(model= best_model, test_dataloader= test_loader)
print (real_acc)
            




