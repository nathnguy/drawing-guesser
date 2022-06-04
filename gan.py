# gan.py
# Generative Adversarial Network

from typing import runtime_checkable
import torch
import torchvision
import torchvision.transforms as transforms
import math
from torch import nn
from discriminator import Discriminator
from generator import Generator
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import skimage

train_data_length = 1024

torch.manual_seed(111)
drawings = np.load('data/duck.npy').astype('float32')
drawings = drawings[:train_data_length]
drawings /= 255.0
drawings -= 0.5
drawings /= 0.5

# train_data  = torch.tensor([np.reshape(x, (-1, 28)) for x in drawings])
train_data = torch.tensor(drawings)
train_labels = torch.zeros(train_data_length)

train_set = TensorDataset(train_data, train_labels)

batch_size = 32
 
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

# use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train
discriminator = Discriminator().to(device=device)
generator = Generator().to(device=device)

# parameters
lr = 0.00005
num_epochs = 100
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(
            device=device
        )
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device
        )
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(
            device=device
        )
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        # output_discriminator = discriminator(all_samples)
        output_discriminator = discriminator(torch.tensor([skimage.util.random_noise(x.detach().numpy(), mode='gaussian', mean=0, var=0.05, clip=True).astype('float32') for x in all_samples]))
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device
        )

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

torch.save(generator.state_dict(), 'model/generator.pth')
