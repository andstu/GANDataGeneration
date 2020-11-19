import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torchvision import transforms, datasets
import numpy as np
import os
from lib.DataCreationWrapper import *


def get_unbalanced_mnist(class_weights, batch_size=100):
    compose = transforms.Compose(
        [
            transforms.ToTensor(),
            # AddNormalNoise(0, .2),
            transforms.Normalize((.5,),(.5))
        ])
    path = "lib/datasets"
    mnist = datasets.MNIST(root=path, train=True, transform=compose, download=True)
    
    samples_weight = [class_weights[t] for t in mnist.targets]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    batch_sampler = BatchSampler(sampler, batch_size, False)

    return DataLoader(mnist, batch_size=batch_size, sampler=sampler)

# NOISE TRANSFORMS
class AddNormalNoise(object):
    def __init__(self, mean=0, std=1):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        return sample + torch.randn(sample.size()) * self.std + self.mean

class Noisifier():
    def __init__(self, num_p_comp = 2):
        directory = "pca/"

        pca_data = {}
        for filename in os.listdir(directory):
            label = filename[0]
            data = np.load(directory + filename)

            if(filename[2] == "v"):
                scores = data[:num_p_comp+1] / np.sum(data[:num_p_comp+1])
                probs = np.exp(scores) / np.sum(np.exp(scores))
                pca_data[str(label) + "p"] = torch.from_numpy(probs)
                if torch.cuda.is_available():
                    pca_data[str(label) + "p"].cuda()
            else:
                pca_data[str(label)] = torch.from_numpy(data[0:num_p_comp,:])
                if torch.cuda.is_available():
                    pca_data[str(label)].cuda()
        self.pca_data = pca_data


    def add_noise_random(self, data):
        return data + gaussian_noise(data.size(0), data.size(1), mean = 0, stddev = 1)

    def add_noise_directed(self, data, labels, scale = 1):
        noise_matrix = torch.zeros_like(data)
        i = 0
        for l in labels:
            noise = (scale * self.pca_data["{}p".format(l)] * self.pca_data[str(l)].T).T.sum(axis=0)
            noise_matrix[i] += noise.squeeze()
            i += 1

        return data + noise_matrix


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

def data_loader_to_tensor(data_loader):
    X = None
    Y = None
    for x, y in data_loader:
        if X is None:
            X = x
            Y = y
        else:
            X = torch.cat((X, x))
            Y = torch.cat((Y, y))
    return X, Y