import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

torch.manual_seed(42)
def get():

    full_train = datasets.MNIST(
        root= "./data",
        train= True,
        download= True,
    )

    test_dataset = datasets.MNIST(
        root= "./data",
        train= False,
        download = True,
    )

    train_dataset, val_dataset = random_split(
        full_train,
        [55000, 5000],
        generator= torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size= 128, 
        shuffle= True, 
        num_workers= 0
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size= 128, 
        shuffle= False,
        num_workers= 0
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size= 128, 
        shuffle= False,
        num_workers= 0
    )
    return train_loader, val_loader, test_loader

