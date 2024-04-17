import numpy as np
import random
import pickle
from collections import deque
from model import QNetwork
from replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F 
import torch.optim as optim

BUFFER_SIZE = 100000
BATCH_SIZE = 1024
LR = 0.00005

base_dir = './data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size=4*4, action_size=4, seed=42, 
                 fc1_units=256, fc2_units=256, fc3_units=256,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 lr=LR, use_expected_rewards=True, predict_steps=2):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.batch_size = batch_size
        self.losses = []
        self.use_expected_rewards = use_expected_rewards
        self.current_iteration = 0
        
        # game scores graph
        self.last_n_scores = deque(maxlen=50)
        self.max_score = 0
        self.best_score_board = []
        
        # mean rewards graph
        self.last_n_total_rewards = deque(maxlen=50)
        self.mean_total_rewards = []
        
        # max cell value on game board
        self.max_vals_list = []
        self.best_val_board = []
        self.max_val = 0
        
        # num of steps per episode
        self.max_steps_list = []
        self.last_n_steps = deque(maxlen=50)
        self.mean_steps = []
        self.max_steps = 0
        self.total_steps = 0
        self.best_steps_board = []
        
        # q net
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units, fc3_units=fc3_units).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units, fc3_units=fc3_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9999)
        
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        
        self.t_step = 0
        self.steps_ahead = predict_steps
        
    def save(self, name):
        torch.save(self.qnetwork_local.state_dict(), base_dir+'/network_local_%s.pth' % name)
        torch.save(self.qnetwork_target.state_dict(), base_dir+'/network_target_%s.pth' % name)
        torch.save(self.optimizer.state_dict(), base_dir+'/optimizer_%s.pth' % name)
        torch.save(self.lr_decay.state_dict(), base_dir+'/lr_schd_%s.pth' % name)
            
    def load(self, name):
        self.qnetwork_local.load_state_dict(torch.load(base_dir+'/network_local_%s.pth' % name))
        self.qnetwork_target.load_state_dict(torch.load(base_dir+'/network_target_%s.pth' % name))
        self.optimizer.load_state_dict(torch.load(base_dir + '/optimizer_%s.pth' % name))
        self.lr_decay.load_state_dict(torch.load(base_dir + '/lr_schd_%s.pth' % name))
        
    def step(self, state, action, reward, next_state, done, error):
        self.memory.add(state, action, reward, next_state, done, error)
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        return action_values.cpu().data.numpy()
    
    def learn(self, learn_iterations, save_loss=True):
        if self.use_expected_rewards:
            self.memory.calc_expected_rewards(self.steps_ahead)
            
        self.memory.add_episode_experience()
        
        losses = []
        
        if len(self.memory) > self.batch_size:
            if learn_iterations is None:
                learn_iterations = self.learn_iterations
                
            for _ in range(learn_iterations):
                
                states, actions, rewards, _, _ = self.memory.sample()
                
                Q_expected = self.qnetwork_local(states).gather(1, actions)
                
                loss = F.mse_loss(Q_expected, rewards)
                
                losses.append(loss.detach().cpu().numpy())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.lr_decay.step()
            
            if save_loss:
                self.losses.append(np.mean(losses))
        
        else:
            self.losses.append(0)