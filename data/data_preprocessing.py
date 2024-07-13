import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader

def augment_data(train_set):
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    augmented_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=augment_transform)
    return augmented_train_set

def split_data(train_set):
    train_data, val_data = train_test_split(train_set, test_size=0.2, random_state=42)
    return train_data, val_data
