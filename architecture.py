# architecture.py
# Our neural network architecture! Aimed at classifying 28x28 drawing data

import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleNet(nn.Module):
    def __init__(self, inputs=28*28, hidden=512, outputs=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)  # activaction function: 92.4% -> 98.6% accuracy
        x = self.fc2(x)
        return x