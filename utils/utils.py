import torch

def calculate_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    
    return mean, std
