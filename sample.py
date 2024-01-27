import os
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from model.model_mini import Generator, Discriminator, noise_dim, password_length

def load_model(model_path, netG, netD, optimizerG, optimizerD):
    checkpoint = torch.load(model_path)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

netG = Generator(password_length).cuda()
netD = Discriminator(password_length).cuda()
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()
model_path = 'runs/out_model/model_epoch_10.ckpt'
epoch, loss = load_model(model_path, netG, netD, optimizerG, optimizerD)

def generate_password(netG, noise_dim):
    noise = torch.randn(1, noise_dim, device="cuda")
    with torch.no_grad():
        netG.eval()
        generated_data = netG(noise)
    generated_password = ''.join([chr(int(c)) for c in generated_data[0]])
    return generated_password
password = generate_password(netG, noise_dim)
print("Generated Password:", password)
