import math
import random
from state import State
from model import DQN

import torch
import torch.nn as nn
import torch.optim as optim

class Agent:
    def __init__(self, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, replay_buffer_size=50000, batch_size=128, target_update_fequency=5000, mem_discount=0.9):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.target_update_frequency = target_update_fequency
        self.mem_discount = mem_discount