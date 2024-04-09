import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=256):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1  = nn.Linear(state_size, fc1_units)
        self.bn1  = nn.BatchNorm1d(fc1_units)
        self.act1 = nn.ReLU()
        self.fc2  = nn.Linear(fc1_units, fc2_units)
        self.bn2  = nn.BatchNorm1d(fc2_units)
        self.act2 = nn.ReLU()
        self.fc3  = nn.Linear(fc2_units, fc3_units)
        self.bn3  = nn.BatchNorm1d(fc3_units)
        self.act3 = nn.ReLU()
        self.fcout= nn.Linear(fc3_units, action_size)
        
    def forward(self, state):
        
        x = self.act1(self.bn1(self.fc1(state)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.fcout(x)
        return x