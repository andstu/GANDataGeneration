import torch
from torch import nn
from lib.DataCreationWrapper import *

# References
# https://github.com/soumith/ganhacks <-- real useful
# https://github.com/diegoalejogm/gans
# https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

# NETWORKS
class DiscriminatorNetwork(torch.nn.Module):
    def __init__(self, num_input_features, num_classes):
        super(DiscriminatorNetwork, self).__init__()
        self.n_input_features = num_input_features

        def block(input_size, output_size):
            return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )  
        
        # self.hidden0 = nn.Sequential(
        #     nn.Linear(num_input_features, 32),
        #     nn.LeakyReLU(0.2),
        #     nn.BatchNorm1d(32)
        # )

        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.preprocess = nn.Sequential(
            nn.Flatten()
        )
        self.hidden0 = block(num_input_features + num_classes, 1024)
        self.hidden1 = block(1024, 512)
        self.hidden2 = block(512, 256)

        self.out = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = self.preprocess(x)
        x = torch.cat((x,self.label_embedding(labels)),-1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

# Regular Generator Network    
class GeneratorNetwork(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, num_classes):
        super(GeneratorNetwork, self).__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features

        def block(input_size, output_size):
            return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.2)
        )

        # self.hidden0 = nn.Sequential(
        #     nn.Linear(num_input_features, 128),
        #     nn.LeakyReLU(0.2),
        #     nn.BatchNorm1d(128),
        #     nn.Dropout(0.3)
        # )

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.hidden0 = block(num_input_features + num_classes, 256)
        self.hidden1 = block(256, 512)
        self.hidden2 = block(512, 1024)

        
        self.out = nn.Sequential(
            nn.Linear(1024, num_output_features),
            nn.Tanh()
        )
    
    def forward(self, x, labels):
        x = torch.cat((x,self.label_embedding(labels)),-1)
        x  = self.hidden0(x)
        x  = self.hidden1(x)
        x  = self.hidden2(x)
        x = self.out(x)
        return x



    
# Training Functions
def train_discriminator(discr_nn, discr_optimizer, loss, gen_nn, real_data, noise_function, n_classes, labels):    
    # Makes Fake Data    
    batch_size = real_data.size(0)
    fake_data, fake_labels = synthesize_data_and_labels(gen_nn, batch_size, noise_function, n_classes)

    # Zero Grad
    discr_optimizer.zero_grad()

    
    # Prediction On Fake Data     
    fake_discr_pred = discr_nn(fake_data, fake_labels)
    fake_loss = loss(fake_discr_pred, fake_target(batch_size, 0.01))
    fake_loss.backward()
    
    # Prediction On Real Data     
    real_discr_pred = discr_nn(real_data, labels)
    real_loss = loss(real_discr_pred, real_target(batch_size, 0.01))
    real_loss.backward()
    
    discr_optimizer.step()
    
    return real_loss, fake_loss

def train_generator(gen_nn, gen_optimizer, loss, discr_nn, real_data, noise_function, n_classes):
    # Makes Fake Data
    batch_size = real_data.size(0)
    fake_data, fake_labels = synthesize_data_and_labels(gen_nn, batch_size, noise_function, n_classes)

    # Zero Grad
    gen_optimizer.zero_grad()
    
    # Prediction On Fake Data     
    fake_discr_pred = discr_nn(fake_data, fake_labels)
    gen_loss = loss(fake_discr_pred, real_target(batch_size, 0.01)) # Maximizing as opposed to minimizeing
    gen_loss.backward()
    # print("gen_loss loss", gen_loss.grad)
    
    gen_optimizer.step()
    
    return gen_loss

def train_discriminator_wass(discr_nn, discr_optimizer, loss, gen_nn, real_data, noise_function, n_classes, labels):
    # Zero Grad
    discr_optimizer.zero_grad()
    
    # Makes Fake Data    
    batch_size = real_data.size(0)
    fake_data, fake_labels = synthesize_data_and_labels(gen_nn, batch_size, noise_function, n_classes)
    
    # Prediction on Fake Data
    prediction_fake = discr_nn(fake_data, fake_labels)
    
    # Prediction on Real Data
    prediction_real = discr_nn(real_data, labels)
    
    real_loss = - torch.mean(prediction_real)
    fake_loss = torch.mean(prediction_fake)
    
    loss = real_loss + fake_loss
    loss.backward()
    discr_optimizer.step()
    
    return real_loss, fake_loss

def train_generator_wass(gen_nn, gen_optimizer, loss, discr_nn, real_data, noise_function, n_classes):
    # Zero Grad
    gen_optimizer.zero_grad()
    
    # Makes Fake Data
    batch_size = real_data.size(0)
    fake_data, fake_labels = synthesize_data_and_labels(gen_nn, batch_size, noise_function, n_classes)
    # fake_data = synthesize_data(gen_nn, batch_size, noise_function)
        
    # Prediction on Fake Data
    prediction_fake = discr_nn(fake_data, fake_labels)
    
    loss = - torch.mean(prediction_fake)
    loss.backward()
    gen_optimizer.step()
    
    return loss
