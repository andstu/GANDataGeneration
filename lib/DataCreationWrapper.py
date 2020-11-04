import torch
from torch.autograd.variable import Variable

# TARGET FUNCTIONS
def real_target(size):
    target = Variable(torch.ones(size, 1))
    if torch.cuda.is_available():
        return target.cuda()
    return target

def fake_target(size):
    target = Variable(torch.zeros(size, 1))
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