import torch
from torch import nn
from lib.DataCreationWrapper import *

# References
# https://github.com/soumith/ganhacks <-- real useful
# https://github.com/diegoalejogm/gans
# https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

# NETWORKS
class DiscriminatorNetwork(torch.nn.Module):
    def __init__(self, num_input_features):
        super(DiscriminatorNetwork, self).__init__()
        self.n_input_features = num_input_features
        
        self.hidden0 = nn.Sequential(
            nn.Linear(num_input_features, 32),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(32,64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(64,128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128)
        )
        
        self.hidden3 = nn.Sequential(
            nn.Linear(128,64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64)
        )

        self.hidden4 = nn.Sequential(
            nn.Linear(64,32),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32)
        )

        self.hidden5 = nn.Sequential(
            nn.Linear(32,32),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32)
        )

        self.hidden6 = nn.Sequential(
            nn.Linear(32,16),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16)
        )
        
        self.out = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x  = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.out(x)
        return x

# Regular Generator Network    
class GeneratorNetwork(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(GeneratorNetwork, self).__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features

        self.hidden0 = nn.Sequential(
            nn.Linear(num_input_features, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

        self.hidden3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )

        self.hidden4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )

        self.hidden5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)
        )

        self.hidden6 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)
        )
        
        self.out = nn.Sequential(
            nn.Linear(32, num_output_features),
            nn.Tanh()
        )
    
    def forward(self, x):
        x  = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.out(x)
        return x



    
# Training Functions
def train_discriminator(discr_nn, discr_optimizer, loss, gen_nn, real_data, noise_function):    
    # Makes Fake Data    
    batch_size = real_data.size(0)
    fake_data = synthesize_data(gen_nn, batch_size, noise_function)

    # Zero Grad
    discr_optimizer.zero_grad()

    
    # Prediction On Fake Data     
    fake_discr_pred = discr_nn(fake_data)
    fake_loss = loss(fake_discr_pred, fake_target(batch_size, 0.01))
    fake_loss.backward()
    
    # Prediction On Real Data     
    real_discr_pred = discr_nn(real_data)
    real_loss = loss(real_discr_pred, real_target(batch_size, 0.01))
    real_loss.backward()
    
    discr_optimizer.step()
    
    return fake_loss + real_loss

def train_generator(gen_nn, gen_optimizer, loss, discr_nn, real_data, noise_function):
    # Makes Fake Data
    batch_size = real_data.size(0)
    fake_data = synthesize_data(gen_nn, batch_size, noise_function)

    # Zero Grad
    gen_optimizer.zero_grad()
    
    # Prediction On Fake Data     
    fake_discr_pred = discr_nn(fake_data)
    gen_loss = loss(fake_discr_pred, real_target(batch_size, 0.01)) # Maximizing as opposed to minimizeing
    gen_loss.backward()
    # print("gen_loss loss", gen_loss.grad)
    
    gen_optimizer.step()
    
    return gen_loss
