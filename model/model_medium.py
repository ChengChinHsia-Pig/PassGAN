import torch
import torch.nn as nn

# 50 million

class Generator(nn.Module):
    def __init__(self, password_length):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(8192),
            nn.Linear(8192, 16384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(16384),
            nn.Linear(16384, password_length)
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, password_length):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(password_length, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8192, 16384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16384, 8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
        
    def forward(self, password):
        return self.main(password)

noise_dim = 100
