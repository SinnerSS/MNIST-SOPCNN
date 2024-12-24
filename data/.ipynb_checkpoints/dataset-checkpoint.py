import torch
from torchvision import datasets, transforms

def get_train_loader(batch_size, data_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # # Debug: Print a batch of data
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    # print(f'Train images batch shape: {images.shape}')
    # print(f'Train labels batch shape: {labels.shape}')
    
    return train_loader

def get_test_loader(batch_size, data_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # # Debug: Print a batch of data
    # data_iter = iter(test_loader)
    # images, labels = next(data_iter)
    # print(f'Test images batch shape: {images.shape}')
    # print(f'Test labels batch shape: {labels.shape}')
    
    return test_loader
