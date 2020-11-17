import torch
from torch.autograd.variable import Variable

# TARGET FUNCTIONS
def make_target(size, value):
    target = Variable(torch.zeros(size, 1) + value)
    if torch.cuda.is_available():
        return target.cuda()
    return target

# NOISE FUNCTIONS

def uniform_noise(num_elements, num_features):
    noise = torch.rand(num_elements, num_features)
    if torch.cuda.is_available():
        return noise.cuda()
    return noise

def gaussian_noise(num_elements, num_features, mean = 0, stddev = 1):
    noise = torch.empty(num_elements, num_features).normal_(mean, stddev)
    if torch.cuda.is_available():
        return noise.cuda()
    return noise

# SYNTHESIS FUNCTIONS 

def synthesize_data(gen_nn, batch_size, noise_function):
    noise = noise_function(batch_size, gen_nn.num_input_features)
    fake_data = gen_nn(noise)
    return fake_data

def synthesize_data_and_labels(gen_nn, batch_size, noise_function, n_classes):
    noise = noise_function(batch_size, gen_nn.num_input_features)
    labels = torch.randint(0,n_classes, (batch_size,))
    if torch.cuda.is_available():
        labels = labels.cuda()

    fake_data = gen_nn(noise, labels)
    return fake_data, labels

def synthesize_data_from_label(gen_nn, batch_size, noise_function, label):
    noise = noise_function(batch_size, gen_nn.num_input_features)
    labels = torch.full((batch_size,), label)
    if torch.cuda.is_available():
        labels = labels.cuda()

    fake_data = gen_nn(noise, labels)
    return fake_data

def synthesize_data_from_each_label(gen_nn, noise_function, n_classes):
    noise = noise_function(n_classes, gen_nn.num_input_features)
    labels = torch.arange(n_classes)
    if torch.cuda.is_available():
        labels = labels.cuda()

    fake_data = gen_nn(noise, labels)
    return fake_data