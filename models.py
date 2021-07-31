import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.max_pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(32, 16)
        self.linear_2 = nn.Linear(16, 1)
        
    def conv(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        
        return x
        
    def ffnn(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        
        return x
    
    def forward(self, x):
        x = self.conv(x)
        x = self.ffnn(x)
        
        return x
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)