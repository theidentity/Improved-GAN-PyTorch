import torch
import torch.nn as nn
import torch.functional as F
import helpers


def get_cifar_gan_networks(latent_dim,num_classes):

    class Discriminator(nn.Module):
        """docstring for Discriminator"""
        def __init__(self,num_classes):
            super(Discriminator, self).__init__()
            self.net = nn.Sequential(
                    nn.Dropout(.2),
                    nn.Conv2d(3,96,3,stride=1,padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(96,96,3,stride=1,padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(96,96,3,stride=2,padding=1),
                    nn.LeakyReLU(),

                    nn.Dropout(.5),
                    nn.Conv2d(96,192,3,stride=1,padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(192,192,3,stride=1,padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(192,192,3,stride=2,padding=1),
                    nn.LeakyReLU(),
                    
                    nn.Dropout(.5),
                    nn.Conv2d(192,192,3,stride=1,padding=0),
                    nn.LeakyReLU(),
                    nn.Conv2d(192,192,1,stride=1,padding=0),
                    nn.LeakyReLU(),
                    nn.Conv2d(192,192,1,stride=1,padding=0),
                    nn.LeakyReLU(),

                    nn.MaxPool2d(6,stride=1),
                    helpers.Flatten()
                )

            self.fc = nn.Linear(192,num_classes)
            
        def forward(self,x):
            inter_layer = self.net(x)
            logits = self.fc(inter_layer)
            return inter_layer, logits

    class Generator(nn.Module):
        """docstring for Generator"""
        def __init__(self,latent_dim):
            super(Generator, self).__init__()
            self.net = nn.Sequential(
                    nn.Linear(latent_dim,512*4*4),
                    nn.BatchNorm1d(512*4*4),
                    nn.ReLU(),
                    helpers.Reshape((512,4,4)),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
                    nn.Tanh(),
                )

        def forward(self,x):
            return self.net(x)

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight,mean=.0,std=.1)
            nn.init.constant_(m.bias,.0)

        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,mean=0,std=.05)

    G = Generator(latent_dim=latent_dim).apply(init_normal)
    D = Discriminator(num_classes=num_classes).apply(init_normal)
    return G,D

def get_mnist_gan_networks(latent_dim,num_classes):

    class Discriminator(nn.Module):
        """docstring for Discriminator"""
        def __init__(self,num_classes):
            super(Discriminator, self).__init__()
            self.net = nn.Sequential(
                    nn.Conv2d(1,64,3,stride=1,padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(64,96,3,stride=2,padding=1),
                    nn.LeakyReLU(),

                    nn.Dropout(.2),
                    nn.Conv2d(96,96,3,stride=1,padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(96,192,3,stride=2,padding=1),
                    nn.LeakyReLU(),

                    nn.Dropout(.2),
                    nn.Conv2d(192,192,3,stride=2,padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(192,192,1,stride=1,padding=0),
                    nn.LeakyReLU(),
                    nn.Conv2d(192,192,1,stride=1,padding=0),
                    nn.LeakyReLU(),

                    nn.MaxPool2d(4,stride=1),
                    helpers.Flatten()
                )

            self.fc = nn.Linear(192,num_classes)
            
        def forward(self,x):
            inter_layer = self.net(x)
            logits = self.fc(inter_layer)
            return inter_layer, logits

    class Generator(nn.Module):
        """docstring for Generator"""
        def __init__(self,latent_dim):
            super(Generator, self).__init__()
            self.net = nn.Sequential(
                    nn.Linear(latent_dim,512*4*4),
                    nn.BatchNorm1d(512*4*4),
                    nn.ReLU(),
                    helpers.Reshape((512,4,4)),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
                    nn.Tanh(),
                )

        def forward(self,x):
            return self.net(x)

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight,mean=.0,std=.1)
            nn.init.constant_(m.bias,.0)

        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,mean=0,std=.05)

    G = Generator(latent_dim=latent_dim).apply(init_normal)
    D = Discriminator(num_classes=num_classes).apply(init_normal)
    return G,D


def get_mnist_linear_networks(latent_dim,num_classes):

    class Generator(nn.Module):
        def __init__(self,latent_dim):
            super(Generator,self).__init__()
            self.net = nn.Sequential(
                    helpers.Flatten(),
                    nn.Linear(latent_dim,500),
                    nn.Softplus(),
                    nn.BatchNorm1d(500),
                    nn.Linear(500,500),
                    nn.Softplus(),
                    nn.BatchNorm1d(500),
                    nn.Linear(500,32*32),
                    nn.Tanh(),
                    helpers.Reshape((1,32,32))
                )

        def forward(self,x):
            return self.net(x)

    class Discriminator(nn.Module):
        def __init__(self,num_classes):
            super(Discriminator, self).__init__()
            self.net = nn.Sequential(
                    helpers.Flatten(),
                    nn.Linear(32*32,1000),
                    nn.Dropout(.1),
                    nn.ReLU(),
                    nn.Linear(1000,500),
                    nn.Dropout(.1),
                    nn.ReLU(),
                    nn.Linear(500,250),
                    nn.Dropout(.1),
                    nn.ReLU(),
                    nn.Linear(250,250),
                    nn.Dropout(.1),
                    nn.ReLU(),
                    nn.Linear(250,250),
                    nn.Dropout(.1),
                    nn.ReLU(),
                    nn.Linear(250,num_classes),
                )

        def forward(self,x):
            return self.net(x)

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight,mean=.0,std=.1)
            nn.init.constant_(m.bias,.0)

    G = Generator(latent_dim).apply(init_normal)
    D = Discriminator(num_classes).apply(init_normal)
    return G,D

if __name__ == '__main__':
    # G,D = get_cifar_gan_networks(latent_dim=100,num_classes=10)
    # G,D = get_mnist_gan_networks(latent_dim=100,num_classes=10)

    # z = torch.randn(20,100)
    # print(G(z).shape)
    # inter_layer,logits = D(G(z))
    # print(inter_layer.shape)
    # print(logits.shape)
    
    G,D = get_mnist_linear_networks(latent_dim=100,num_classes=10)
    z = torch.randn(20,100)
    print(G(z).shape)
    print(D(G(z)).shape)