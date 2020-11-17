import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torchvision import transforms, datasets
import numpy as np

def get_unbalanced_mnist(class_weights, batch_size=100):
    compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((.5,),(.5))
        ])
    path = "lib/datasets"
    mnist = datasets.MNIST(root=path, train=True, transform=compose, download=True)
    
    samples_weight = [class_weights[t] for t in mnist.targets]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    batch_sampler = BatchSampler(sampler, batch_size, False)

    return DataLoader(mnist, batch_size=batch_size, sampler=sampler)

def format_to_image(imgs, num_imgs, width):
    result = imgs.reshape(num_imgs,width,width)
    result = (result + 1) / 2
    return result

# computes the average image for a supplied class
def get_avg_img(data_loader, class_to_mimic, width):
    avg_class_img = torch.zeros(width, width)
    n_class_imgs = 0
    for n_batch, (batch, labels) in enumerate(data_loader):
        valid_sample = batch[labels == class_to_mimic]
        avg_class_img += torch.sum(valid_sample,0).squeeze()
        n_class_imgs += len(valid_sample)
    return format_to_image(avg_class_img / n_class_imgs, 1, width)

# Gets at least num_samples samples
def get_sample(data_loader, num_samples, class_to_mimic, feature_0, feature_1):
    result = None
    n_class_imgs = 0
    for n_batch, (batch, labels) in enumerate(data_loader):
        valid_sample = batch[labels == class_to_mimic]
        if result == None:
            result = valid_sample
        else:
            result = torch.cat((result,valid_sample),0)

        n_class_imgs += len(valid_sample)
        if n_class_imgs > num_samples:
            break

    return torch.reshape(result, (n_class_imgs, -1))