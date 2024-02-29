import torch
import torch.nn as nn
import torch.nn.functional as f

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.inputLayer   = nn.Linear(input_size, 256)
        self.hiddenLayer1 = nn.Linear(256, 256)
        self.hiddenLayer2 = nn.Linear(256, 256)
        self.outputLayer  = nn.Linear(256, output_size)

    def forward(self, x):
        x = f.relu(self.inputLayer(x))
        x = f.relu(self.hiddenLayer1(x))
        x = f.relu(self.hiddenLayer2(x))
        return self.outputLayer(x)