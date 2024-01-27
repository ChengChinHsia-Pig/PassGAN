import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from model.model_mini import Generator, Discriminator, noise_dim, password_length

class PasswordDataset(Dataset):
    def __init__(self, file_path, max_length):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = file.readlines()
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        password = self.data[idx].strip()
        if len(password) > self.max_length:
            password = password[:self.max_length]
        padded_password = password.ljust(self.max_length)
        return torch.tensor([ord(c) for c in padded_password], dtype=torch.float32)

print("Loading Datas")
device = torch.device("cuda:0")
dataset = PasswordDataset('datas/rockyou.txt', password_length)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

netG = Generator(password_length).to(device)
netD = Discriminator(password_length).to(device)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Current training device: {device}")
print(f"Number of parameters in Generator: {count_parameters(netG)}")
print(f"Number of parameters in Discriminator: {count_parameters(netD)}")

writer = SummaryWriter('runs/logs')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

fixed_noise = torch.randn(1, noise_dim, device=device)

def generate_password(netG):
    with torch.no_grad():
        netG.eval()
        generated_data = netG(fixed_noise)
        netG.train()
    generated_password = ''.join([chr(int(c)) for c in generated_data[0]])

    return generated_password

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

netG.apply(weights_init)
netD.apply(weights_init)

save_dir = 'runs/out_model'
os.makedirs(save_dir, exist_ok=True)

print("Training")
for epoch in range(1000000):
    print(f"Epoch:{epoch + 1}")
    start_time = time.time()
    running_loss = 0.0

    current_lr = get_lr(optimizerG)
    writer.add_scalar('Learning Rate', current_lr, epoch)
    
    for i, data in enumerate(dataloader, 0):
        real_passwords = data.to(device)
        netD.zero_grad()
        real_labels = torch.ones(data.size(0), 1).to(device)
        output = netD(real_passwords).view(-1)
        lossD_real = criterion(output, real_labels.view(-1))

        noise = torch.randn(data.size(0), 100, device=device)
        fake_passwords = netG(noise)

        fake_labels = torch.zeros(data.size(0), 1).to(device)
        output = netD(fake_passwords.detach()).view(-1)
        lossD_fake = criterion(output, fake_labels.view(-1))

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
        netG.zero_grad()
        output = netD(fake_passwords).view(-1)
        lossG = criterion(output, real_labels.view(-1))
        lossG.backward()
        optimizerG.step()

        total_loss = lossD + lossG
        running_loss += total_loss.item()

    if epoch+1 % 10 == 0:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'loss': total_loss,
        }, os.path.join(save_dir, f'model_epoch_{epoch+1}.ckpt'))
        print(f"Model Saved, CheckPoint:{epoch+1}")
    
    writer.add_scalar('Training Loss', running_loss / len(dataloader), epoch)
    writer.add_scalar('Epoch Training Time', time.time() - start_time, epoch)
    
    print(f"Epoch:{epoch + 1}, {time.time() - start_time}s.")
    writer.close()

writer.close()
print("Train Completed.")