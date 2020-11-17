import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets

def get_unbalanced_emnist(weights, batch_size=100):
    compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((.5,),(.5))
        ])
    path = "lib/datasets"
    mnist = datasets.MNIST(root=path, train=True, transform=compose, download=True)
    sampler = WeightedRandomSampler(weights, batch_size, replacement=True)
    return DataLoader(mnist, batch_size=batch_size, sampler=sampler, )