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
GAMMA = 0.99
LR = 0.00005
TAU = 0.001

base_dir = './data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size=4*4, action_size=4, seed=42, 
                 fc1_units=256, fc2_units=256, fc3_units=256,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 lr=LR, use_expected_rewards=True, predict_steps=2,
                 gamma=GAMMA, tau=TAU):
        TAU = tau
        GAMMA = gamma
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.batch_size = batch_size
        self.losses = []
        self.use_expected_rewards = use_expected_rewards
        self.current_iteration = 0
        
        # game scores
        self.scores_list = []
        self.last_n_scores = deque(maxlen=50)
        self.mean_scores = []
        self.max_score = 0
        self.min_score = 1000
        self.best_score_board = []
        
        # rewards
        self.total_rewards_list = []
        self.last_n_total_rewards = deque(maxlen=50)
        self.mean_total_rewards = []
        self.max_total_reward = 0
        self. best_reward_board = []
        
        # max cell value on game board
        self.max_vals_list = []
        self.last_n_vals = deque(maxlen=50)
        self.mean_vals = []
        self.max_val = 0
        self.best_val_board = []
        
        # num of steps per episode
        self.max_steps_list = []
        self.last_n_steps = deque(maxlen=50)
        self.mean_steps = []
        self.max_steps = 0
        self.total_steps = 0
        self.best_steps_board = []
        
        self.actions_avg_list = []
        self.actions_deque = {
            0:deque(maxlen=50),
            1:deque(maxlen=50),
            2:deque(maxlen=50),
            3:deque(maxlen=50)
        }
        
        # q net
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units, fc3_units=fc3_units).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units, fc3_units=fc3_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        lr_s = lambda epoch: 0.998 ** (epoch % 1000) if epoch < 100000 else 0.999 ** (epoch % 1000)
        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9999)
        
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        
        self.t_step = 0
        self.steps_ahead = predict_steps
        
    def save(self, name):
        torch.save(self.qnetwork_local.state_dict(), base_dir+'/network_local_%s.pth' % name)
        torch.save(self.qnetwork_target.state_dict(), base_dir+'/network_target_%s.pth' % name)
        torch.save(self.optimizer.state_dict(), base_dir+'/optimizer_%s.pth' % name)
        torch.save(self.lr_decay.state_dict(), base_dir+'/lr_schd_%s.pth' % name)
        state = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'losses': self.losses,
            'use_expected_rewards': self.use_expected_rewards,
            'current_iteration': self.current_iteration,
        
        # Game scores
            'scores_list': self.scores_list,
            'last_n_scores': self.last_n_scores,
            'mean_scores': self.mean_scores,
            'max_score': self.max_score,
            'min_score': self.min_score,
            'best_score_board': self.best_score_board,

        # Rewards
            'total_rewards_list': self.total_rewards_list,
            'last_n_total_rewards': self.last_n_total_rewards,
            'mean_total_rewards': self.mean_total_rewards,
            'max_total_reward': self.max_total_reward,
            'best_reward_board': self.best_reward_board,

        # Max cell value on game board
            'max_vals_list': self.max_vals_list,
            'last_n_vals': self.last_n_vals,
            'mean_vals': self.mean_vals,
            'max_val': self.max_val,
            'best_val_board': self.best_val_board,

        # Number of steps per episode
            'max_steps_list': self.max_steps_list,
            'last_n_steps': self.last_n_steps,
            'mean_steps': self.mean_steps,
            'max_steps': self.max_steps,
            'total_steps': self.total_steps,
            'best_steps_board': self.best_steps_board,
        
            'actions_avg_list': self.actions_avg_list,
            'actions_deque': self.actions_deque,
        # Replay buffer
            'memory': self.memory.dump(),
        # Initialize time step
            't_step': self.t_step,
            'steps_ahead': self.steps_ahead
        }

        with open(base_dir+'/agent_state_%s.pkl' % name, 'wb') as f:
            pickle.dump(state, f)
            
    def load(self, name):
        self.qnetwork_local.load_state_dict(torch.load(base_dir+'/network_local_%s.pth' % name))
        self.qnetwork_target.load_state_dict(torch.load(base_dir+'/network_target_%s.pth' % name))
        self.optimizer.load_state_dict(torch.load(base_dir + '/optimizer_%s.pth' % name))
        self.lr_decay.load_state_dict(torch.load(base_dir + '/lr_schd_%s.pth' % name))
        
        with open(base_dir+'/agent_state_%s.pkl' % name, 'rb') as f:
            state = pickle.load(f)

        self.state_size = state['state_size']
        self.action_size = state['action_size']
        self.seed = state['seed']
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.batch_size = state['batch_size']
        self.losses = state['losses']
        self.use_expected_rewards = state['use_expected_rewards']
        self.current_iteration = state['current_iteration']
        
        self.scores_list = state['scores_list']
        self.last_n_scores = state['last_n_scores']
        self.mean_scores = state['mean_scores']
        self.max_score = state['max_score']
        self.min_score = state['min_score'] if 'min_score' in state.keys() else state['max_score']
        self.best_score_board = state['best_score_board']

        self.total_rewards_list = state['total_rewards_list']
        self.last_n_total_rewards = state['last_n_total_rewards']
        self.mean_total_rewards = state['mean_total_rewards']
        self.max_total_reward = state['max_total_reward']
        self.best_reward_board = state['best_reward_board']

        self.max_vals_list = state['max_vals_list']
        self.last_n_vals = state['last_n_vals']
        self.mean_vals = state['mean_vals']
        self.max_val = state['max_val']
        self.best_val_board = state['best_val_board']

        self.max_steps_list = state['max_steps_list']
        self.last_n_steps = state['last_n_steps']
        self.mean_steps = state['mean_steps']
        self.max_steps = state['max_steps']
        self.total_steps = state['total_steps']
        self.best_steps_board = state['best_steps_board']
        
        self.actions_avg_list = state['actions_avg_list']
        self.actions_deque = state['actions_deque']

        self.memory.load(state['memory'])
        
        self.t_step = state['t_step']
        self.steps_ahead = state['steps_ahead']
        
    def step(self, state, action, reward, next_state, done, error, action_dist):
        self.memory.add(state, action, reward, next_state, done, error, action_dist, None)
        
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        return action_values.cpu().data.numpy()
    
    def learn(self, learn_iterations, mode='board_max', save_loss=True, gamma=GAMMA, weight=None, bootstrapping=False):
        if self.use_expected_rewards:
            self.memory.calc_expected_rewards(self.steps_ahead, weight)
            
        self.memory.add_episode_experience()
        
        losses = []
        
        if len(self.memory) > self.batch_size:
            if learn_iterations is None:
                learn_iterations = self.learn_iterations
                
            for i in range(learn_iterations):
                
                states, actions, rewards, next_states, done = self.memory.sample(mode=mode)
                
                Q_expected = self.qnetwork_local(states).gather(1, actions)
                
                loss = F.mse_loss(Q_expected, rewards)
                
                if not bootstrapping:
                    losses.append(loss.detach().cpu().numpy())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.lr_decay.step()
            
            if save_loss:
                self.losses.append(np.mean(losses))
        
        else:
            self.losses.append(0)