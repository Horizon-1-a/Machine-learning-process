import numpy as np
import torch
from torch import nn
from data import get
from model import Mymodel
from engine import train_one_model, evaluate, test_phase
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get()
Train_model = Mymodel(256).to(device)
best_model = Mymodel(256).to(device)
optimizer = torch.optim.SGD(Train_model.parameters(), lr = 0.1)
loss_fn = nn.CrossEntropyLoss()
epochs = 30

best_acc = 0.0

train_loss = np.array([])
val_loss = np.array([])


# train 
for i in range (epochs):
    Train_model.train()
    print (f'epoch {i}: ')
    train_epochs = 0.0
    train_num = 0
    val_epochs = 0.0
    val_num = 0
    for x, y_true in train_loader:
        optimizer.zero_grad()
        x, y_true = x.to(device), y_true.to(device)
        z = Train_model(x)
        train_epochs += train_one_model(z, y_true= y_true, loss_fn= loss_fn, optimizer= optimizer)
        train_num += 1
    train_loss.append(train_epochs / train_num)

    #validating 
    with torch.no_grad():
        Train_model.eval()
        res, cur_acc, validate_loss = evaluate(val_dataloader= val_loader, model= Train_model, best_accuracy= best_acc)
        val_loss.append(validate_loss)
        if (res != None):
            best_model.load_state_dict(res)
            best_model.train()
            best_acc = cur_acc

real_acc = test_phase(model= best_model, test_dataloader= test_loader, device= device)
print (real_acc)
            
# Thiết lập biểu đồ
plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, 'bo-', label='Training Loss', linewidth=2)
plt.plot(epochs, val_loss, 'ro--', label='Validation Loss', linewidth=2)

# Tùy chỉnh giao diện
plt.title('Biểu đồ hội tụ Loss qua các Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.show()



