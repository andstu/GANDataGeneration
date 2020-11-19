import torch
from torch import nn
from torch.autograd.variable import Variable
from lib.DataCreationWrapper import *
import numpy as np

# References
# https://github.com/soumith/ganhacks <-- real useful
# https://github.com/diegoalejogm/gans
# https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

# NETWORKS
class DiscriminatorNetwork(torch.nn.Module):
    def __init__(self, num_input_features, num_classes, loss_key):
        super(DiscriminatorNetwork, self).__init__()
        self.n_input_features = num_input_features

        def block(input_size, output_size):
            return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
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

        if(loss_key == 1):
            self.out = nn.Sequential(
                nn.Linear(256, 1)
            )
        else:
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
            nn.BatchNorm1d(output_size, momentum=0.8),
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


class OLD_GeneratorNetwork(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, num_classes):
        super(OLD_GeneratorNetwork, self).__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features

        def block(input_size, output_size):
            return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size, momentum=0.8),
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

# LEFT OFF HERE


class Conv_DiscriminatorNetwork(torch.nn.Module):
    def __init__(self, num_input_features, num_classes, loss_key):
        super(Conv_DiscriminatorNetwork, self).__init__()
        self.width = int(np.sqrt(num_input_features))

        self.input_width = 7
        self.input_channels = 128

        def block(input_channels, output_channels, filter_size, stride, pooling_size=2, pooling_stride=1):
            return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, filter_size, stride=stride, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(pooling_size, stride=pooling_stride)
        )  
   
        self.label_embedding = nn.Sequential(
            nn.Embedding(num_classes, num_input_features)
        )

        self.to_input_form = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_input_features, self.input_channels * self.input_width ** 2),
            nn.LeakyReLU(0.2)
        )

        self.hidden0 = block(self.input_channels,64, 3, 1)
        self.hidden1 = block(64,32, 3, 1)
        
        if(loss_key == 1):
            self.out = nn.Sequential(
                nn.Flatten(),
                nn.Linear(5*5*32, 1)
            )
        else:
            self.out = nn.Sequential(
                nn.Flatten(),
                nn.Linear(5*5*32, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x, labels):
        # print(x.shape)
        # print(self.label_embedding(labels).shape)
        x = self.process_inputs(x, labels)
        # print(x.shape)
        x = self.hidden0(x)
        # print(x.shape)
        x = self.hidden1(x)
        # print(x.shape)
        x = self.out(x)
        return x

    def get_label_embeddings(self, labels):
        return self.label_embedding(labels).view(len(labels), self.width, self.width)

    def process_inputs(self, x, labels):
        return self.to_input_form( x * self.label_embedding(labels).view(len(labels), 1, self.width, self.width)).view(len(labels), self.input_channels, self.input_width, self.input_width)


# Regular Generator Network

#InfoGan
class Conv_GeneratorNetwork(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, num_classes):
        super(Conv_GeneratorNetwork, self).__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        
        self.input_width = 14
        self.input_channels = 128

        self.label_embedding = nn.Sequential(
            nn.Embedding(num_classes, num_input_features)
        )

        self.to_input_form = nn.Sequential(
            nn.Linear(num_input_features, self.input_channels * self.input_width ** 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.input_channels * self.input_width ** 2)
        )

        self.hidden0 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Dropout(.4)
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(64,16,4,stride=2,padding=1),
            nn.LeakyReLU(0.1)
        )
        
        self.out = nn.Sequential(
            nn.Conv2d(16,1,4,stride=2,padding=1),
            nn.LeakyReLU(0.1),
            nn.Tanh()
        )
    
    def forward(self, x, labels):
        x = self.process_inputs(x, labels)
        
        # x = self.process_noise(x).view(len(labels), self.input_channels, self.input_width, self.input_width)
        x  = self.hidden0(x)
        x  = self.hidden1(x)
        x = self.out(x)
        return x

    def process_inputs(self, x, labels):
        return self.to_input_form( x * self.label_embedding(labels) ).view(len(labels), self.input_channels, self.input_width, self.input_width)

    # #visualize
    # def get_label_embeddings(self, labels):
    #     return self.process_inputs(torch.ones_like(labels), labels)

# ARCHIVED
# class Conv_GeneratorNetwork(torch.nn.Module):
#     def __init__(self, num_input_features, num_output_features, num_classes):
#         super(Conv_GeneratorNetwork, self).__init__()
#         self.num_input_features = num_input_features
#         self.num_output_features = num_output_features
        
#         self.input_width = 7
#         self.input_channels = 128

#         self.label_embedding = nn.Sequential(
#             nn.Embedding(num_classes, 50),
#             nn.Linear(50, self.input_width ** 2),
#             nn.LeakyReLU(0.2)
#         )

#         self.process_noise = nn.Sequential(
#             nn.Linear(num_input_features, self.input_channels * (self.input_width ** 2)),
#             nn.LeakyReLU(0.1)
#             # nn.BatchNorm1d(1024),
#             # nn.Linear(1024, self.input_channels * (self.input_width ** 2))
#         )

#         self.hidden0 = nn.Sequential(
#             nn.ConvTranspose2d(129,64,4,stride=2,padding=1),
#             nn.LeakyReLU(0.1),
#             nn.BatchNorm2d(64)
#         )

#         self.hidden1 = nn.Sequential(
#             nn.ConvTranspose2d(64,1,4,stride=2,padding=1),
#             nn.LeakyReLU(0.1)
#         )
        
#         self.out = nn.Sequential(
#             nn.Tanh()
#         )
    
#     def forward(self, x, labels):
#         x = self.process_noise(x).view(len(labels), self.input_channels, self.input_width, self.input_width)
#         x = torch.cat((x,self.get_label_embeddings(labels)),1)
#         x  = self.hidden0(x)
#         x  = self.hidden1(x)
#         x = self.out(x)
#         return x

#     def get_label_embeddings(self, labels):
#         return self.label_embedding(labels).view(len(labels), 1, self.input_width, self.input_width)

def get_visual_embeddings(gen_nn, n_classes):
    labels = Variable(torch.arange(n_classes))
    if torch.cuda.is_available():
        labels = labels.cuda()
    
    embeddings = gen_nn.get_label_embeddings(labels).squeeze()
    embeddings -= embeddings.min(1, keepdim=True)[0]
    embeddings /= embeddings.max(1, keepdim=True)[0]
    return embeddings

    
# Training Functions
def train_discriminator(discr_nn, discr_optimizer, loss, gen_nn, real_data, noise_function, n_classes, labels, real_target, fake_target, loss_key, reg_constant = 10):    
    # Makes Fake Data    
    batch_size = real_data.size(0)
    fake_data, fake_labels = synthesize_data_and_labels(gen_nn, batch_size, noise_function, n_classes)

    # Zero Grad
    discr_optimizer.zero_grad()
    
    # Prediction On Fake Data     
    fake_discr_pred = discr_nn(fake_data, fake_labels)
    fake_loss = loss(fake_discr_pred, fake_target)
    # fake_loss.backward()
    
    # Prediction On Real Data     
    real_discr_pred = discr_nn(real_data, labels)
    real_loss = loss(real_discr_pred, real_target)
    # real_loss.backward()

    loss = real_loss + fake_loss

    reg_loss = 0
    if(loss_key == 1):
        reg_loss = grad_penalty(reg_constant, discr_nn, real_data, fake_data, labels)
        loss += reg_loss
    loss.backward()
    discr_optimizer.step()
    
    return real_loss, fake_loss, reg_loss

# Training Functions
# https://github.com/EmilienDupont/wgan-gp/blob/50361ca47d260f9585f557b84c136c2e417030d1/training.py#L73
def grad_penalty(reg_constant, discr_nn, real_data, fake_data, labels):    
    alpha = torch.rand_like(real_data)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    interpolated_data = alpha * real_data + (1 - alpha) * fake_data
    interpolated_data = Variable(interpolated_data, requires_grad=True)
    if torch.cuda.is_available():
        interpolated_data = interpolated_data.cuda()

    interpolated_discr_pred = discr_nn(interpolated_data, labels)
    gradients = torch.autograd.grad(outputs=interpolated_discr_pred, inputs=interpolated_data,
                               grad_outputs=torch.ones(interpolated_discr_pred.size()).cuda() if torch.cuda.is_available() else torch.ones(
                               interpolated_discr_pred.size()),
                               create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(real_data.size()[0], -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    return reg_constant * ((gradients_norm - 1) ** 2).mean()
    


def train_generator(gen_nn, gen_optimizer, loss, discr_nn, real_data, noise_function, n_classes, real_target):
    # Makes Fake Data
    batch_size = real_data.size(0)
    fake_data, fake_labels = synthesize_data_and_labels(gen_nn, batch_size, noise_function, n_classes)

    # Zero Grad
    gen_optimizer.zero_grad()
    
    # Prediction On Fake Data     
    fake_discr_pred = discr_nn(fake_data, fake_labels)
    gen_loss = loss(fake_discr_pred, real_target) # Maximizing as opposed to minimizeing
    gen_loss.backward()
    # print("gen_loss loss", gen_loss.grad)
    
    gen_optimizer.step()
    
    return gen_loss

def wasserstein_loss(pred,target):
    return torch.mean(pred * target)

# def train_discriminator_wass(discr_nn, discr_optimizer, loss, gen_nn, real_data, noise_function, n_classes, labels):
#     # Zero Grad
#     discr_optimizer.zero_grad()
    
#     # Makes Fake Data    
#     batch_size = real_data.size(0)
#     fake_data, fake_labels = synthesize_data_and_labels(gen_nn, batch_size, noise_function, n_classes)
    
#     # Prediction on Fake Data
#     prediction_fake = discr_nn(fake_data, fake_labels)
    
#     # Prediction on Real Data
#     prediction_real = discr_nn(real_data, labels)
    
#     real_loss = - torch.mean(prediction_real)
#     fake_loss = torch.mean(prediction_fake)
    
#     loss = real_loss + fake_loss
#     loss.backward()
#     discr_optimizer.step()
    
#     return real_loss, fake_loss

# def train_generator_wass(gen_nn, gen_optimizer, loss, discr_nn, real_data, noise_function, n_classes):
#     # Zero Grad
#     gen_optimizer.zero_grad()
    
#     # Makes Fake Data
#     batch_size = real_data.size(0)
#     fake_data, fake_labels = synthesize_data_and_labels(gen_nn, batch_size, noise_function, n_classes)
#     # fake_data = synthesize_data(gen_nn, batch_size, noise_function)
        
#     # Prediction on Fake Data
#     prediction_fake = discr_nn(fake_data, fake_labels)
    
#     loss = - torch.mean(prediction_fake)
#     loss.backward()
#     gen_optimizer.step()
    
#     return loss
