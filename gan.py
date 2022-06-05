# gan.py
# https://www.kaggle.com/code/kmldas/mnist-generative-adverserial-networks-in-pytorch/notebook
# GAN for generating Google QuickDraw drawings

import torch
from torch.utils.data import DataLoader, TensorDataset
from architecture import D, G, latent_size
import torch.nn as nn
from torchvision.utils import save_image
import os
import numpy as np

category = 'parrot'

drawings = np.load('data/' + category + '.npy').reshape(-1, 1, 28, 28).astype('float32')
train_data_length = 16000
drawings = drawings[:train_data_length]
drawings /= 255.0
drawings -= 0.5
drawings /= 0.5

train_data = torch.tensor(drawings)
train_labels = torch.zeros(train_data_length)

train_set = TensorDataset(train_data, train_labels)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

batch_size = 32
lr = 0.0002
# data_loader = DataLoader(mnist, batch_size, shuffle=True)
data_loader = DataLoader(train_set, batch_size, shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

def train_discriminator(images):
    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
        
    # Loss for real images
    outputs = D(images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # Loss for fake images
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Combine losses
    d_loss = d_loss_real + d_loss_fake
    # Reset gradients
    reset_grad()
    # Compute gradients
    d_loss.backward()
    # Adjust the parameters using backprop
    d_optimizer.step()
    
    return d_loss, real_score, fake_score

def train_generator():
    # Generate fake images and calculate loss
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    labels = torch.ones(batch_size, 1).to(device)
    g_loss = criterion(D(fake_images), labels)

    # Backprop and optimize
    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images

sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

sample_vectors = torch.randn(batch_size, latent_size).to(device)

def save_fake_images(index):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    # print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)

# train
def train():
    num_epochs = 300
    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            # Load a batch & transform to vectors
            images = images.reshape(batch_size, -1).to(device)
            
            # Train the discriminator and generator
            d_loss, real_score, fake_score = train_discriminator(images)
            g_loss, fake_images = train_generator()
            
            # Inspect the losses
            if (i+1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                    .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                            real_score.mean().item(), fake_score.mean().item()))
            
        # Sample and save images
        save_fake_images(epoch+1)

    torch.save(D.state_dict(), 'model/d_' + category + '.pth')
    torch.save(G.state_dict(), 'model/g_' + category + '.pth')

if __name__ == '__main__':
    train()

# testing loading saved models
# G.load_state_dict(torch.load('model/g_duck.pth'))
# G.eval()
# save_fake_images(0)

