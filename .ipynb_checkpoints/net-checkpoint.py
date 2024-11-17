'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from training_parameter import training_parameter
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    @autocast()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  
        out = F.relu(out)
        return out
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.downsample = nn.Conv2d(training_parameter.channel,training_parameter.net_size, kernel_size=1, stride=1, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, training_parameter.channel, training_parameter.channel, 2)
        self.layer2 = self.make_layer(ResidualBlock, training_parameter.net_size, training_parameter.net_size, 2)
        self.lstm = nn.LSTMCell(input_size=training_parameter.net_size, hidden_size=training_parameter.net_size)
        self.fc2 = nn.Linear(in_features=7*5, out_features=96)
        self.fc3 = nn.Linear(training_parameter.matrix_size, training_parameter.net_size)
        self.policy_output = nn.Linear(training_parameter.net_size, 3)
        self.value_output = nn.Linear(training_parameter.net_size, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()
        
    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    @autocast()
    def forward(self, x, hidden_state):
        x = x.view(-1)
        x = F.relu(self.fc2(x))
        x, lstm_memory = self.lstm(x, hidden_state)
        hidden_state = (x, lstm_memory)
        policy = self.policy_output(x)
        value = self.value_output(x)
        policy = self.tanh(policy)
        return policy, value, hidden_state
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Usage: net = Net().to(device)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        # Apply batch norm to the pictures
        self.batch_norm = nn.BatchNorm2d(3)

        # followed by 2 layers of convolutional + max pooling
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 32, 5)
        
        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        
        self.fc2 = nn.Linear(256, 128)
        
        self.fc3 = nn.Linear(128, 64)
        
        self.fc4 = nn.Linear(64, 10)
        '''
        
    def forward(self, x):

        '''
        # Apply batch norm
        x = self.batch_norm(x)
        # Apply convolution + pool
        x = self.pool(F.leaky_relu_(self.conv1(x)))
        x = self.pool(F.leaky_relu_(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # Full-connected layers
        x = F.leaky_relu_(self.fc1(x))
        x = F.leaky_relu_(self.fc2(x))
        x = F.leaky_relu_(self.fc3(x))
        # Apply dropout
        x = self.dropout(x)
        # Observe that softmax activation is causing vanishing gradients, so train the weights as category scores directly.
        x = self.fc4(x)
        return x
        '''

