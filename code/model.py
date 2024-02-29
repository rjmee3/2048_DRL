import torch
import torch.nn as nn
import torch.nn.functional as func

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.inputLayer   = nn.Linear(n_observations, 256)
        self.hiddenLayer1 = nn.Linear(256, 256)
        self.hiddenLayer2 = nn.Linear(256, 256)
        self.outputLayer  = nn.Linear(256, n_actions)

    def forward(self, x):
        x = func.relu(self.inputLayer(x))
        x = func.relu(self.hiddenLayer1(x))
        x = self.hiddenLayer2(x)
        return self.outputLayer(x)