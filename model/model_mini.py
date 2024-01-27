import torch
import torch.nn as nn

# 10 million

class Generator(nn.Module):
    def __init__(self, password_length):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, password_length)
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, password_length):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(password_length, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        
    def forward(self, password):
        return self.main(password)

noise_dim = 100
