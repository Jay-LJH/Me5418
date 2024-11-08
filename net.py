import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from parameter import *
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
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  
        out = F.relu(out)
        return out
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = self.make_layer(ResidualBlock, training_parameter.net_size, training_parameter.net_size, 2)
        self.layer2 = self.make_layer(ResidualBlock, training_parameter.net_size, training_parameter.net_size, 2)
        self.lstm = nn.LSTMCell(input_size=training_parameter.net_size, hidden_size=training_parameter.net_size)
        self.fc1 = nn.Linear(in_features=7*5, out_features=training_parameter.net_size)
        self.fc2 = nn.Linear(in_features=training_parameter.net_size, out_features=training_parameter.net_size)
        self.policy_output = nn.Linear(training_parameter.net_size, 3)
        self.value_output = nn.Linear(training_parameter.net_size, 1)
        
    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x, hidden_state):
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x, lstm_memory = self.lstm(x, hidden_state)
        hidden_state = (x, lstm_memory)
        policy = self.policy_output(x)
        value = self.value_output(x)
        policy[0]=torch.tanh(policy[0])
        policy[1]=torch.sigmoid(policy[1])
        policy[2]=torch.sigmoid(policy[2])
        policy[1] =(policy[1]>0.5).float()
        policy[2] =(policy[2]>0.5).float()
        return policy, value, hidden_state