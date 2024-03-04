import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import DQN

class Agent:
    def __init__(self, input_channels, output_size, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(input_channels, output_size).to(self.device)
        self.target_dqn = DQN(input_channels, output_size).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.gamma = gamma