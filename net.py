import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from training_parameter import training_parameter

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
        self.layer1 = self.make_layer(ResidualBlock, training_parameter.net_size, training_parameter.net_size, 2)
        self.layer2 = self.make_layer(ResidualBlock, training_parameter.net_size, training_parameter.net_size, 2)
        self.lstm = nn.LSTM(input_size=training_parameter.net_size, hidden_size=training_parameter.net_size, num_layers=1, batch_first=True)
       # self.fc1 = nn.Linear(training_parameter.vector_size, training_parameter.net_size)
        self.fc2 = nn.Linear(training_parameter.net_size, training_parameter.net_size)
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
        for cell in x:
            cell = np.pad(cell, (0, 7 - len(seq)), mode='constant', constant_values=0)
        x = np.flatten(x)
        x  = x.pad(x, (0, training_parameter.net_size - len(x)), mode='constant', constant_values=0)
        x = self.layer1(x)
        x = self.maxpool(x)
        x, memory = self.lstm(x, hidden_state)
        x = self.fc2(x)
        x = self.fc3(x)
        hidden_state = (x,memory)
        policy = self.policy_output(x)
        value = self.value_output(x)
        policy = self.tanh(policy)
        return policy, value, hidden_state