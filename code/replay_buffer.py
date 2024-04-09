import numpy as np
import random
from collections import namedtuple, deque
import pandas as pd
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        
        self.episode_memory = []
        self.batch_size = batch_size
        
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "error", "action_dist", "weight"])
        self.geomspaces = [np.geomspace(1., 0.5, i) for i in range(1, 10)]
        
    def dump(self):
        d = {
            'action_size': self.action_size,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'geomspaces': self.geomspaces
        }
        
        d['memory'] = [d._asdict() for d in self.memory]
        
        return d
    
    def load(self, d):
        self.action_size = d['action_size']
        self.batch_size = d['batch_size']
        self.seed = d['seed']
        self.geomspaces = d['geomspaces']
        
        for e in d['memory']:
            self.memory.append(self.experience(**e))
            
    def reset_episode_memory(self):
        self.episode_memory = []
            
    def add(self, state, action, reward, next_state, done, error, action_dist, weight=None):
        e = self.experience(state, action, reward, next_state, done, error, action_dist, weight)
        self.episode_memory.append(e)
        
    def add_episode_experience(self):
        self.memory.extend(self.episode_memory)
        self.reset_episode_memory()
        
    def calc_expected_rewards(self, steps_ahead=1, weight=None):
        rewards = [e.reward for e in self.episode_memory if e is not None]
        
        exp_rewards = [np.sum(rewards[i:i+steps_ahead] * self.geomspaces[steps_ahead-1]) for i in range(len(rewards)-steps_ahead)]

        temp_memory = []
        
        for i, e in enumerate(self.episode_memory[:-steps_ahead]):
            t_e = self.experience(e.state, e.action, exp_rewards[i], e.next_state, e.done, e.error, e.action_dist, weight)
            temp_memory.append(t_e)
            
        self.episode_memory = temp_memory
        
    def sample(self, mode='board_max'):
        if mode == 'random':
            experiences = random.sample(self.memory, k=self.batch_size)
            
        elif mode == 'board_max':
            probs = np.array([e.state.max() for e in self.memory])
            probs = probs / probs.sum()
            index = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)
            for i in index:
                experiences.append(self.memory[i])
                
        elif mode == 'board_sum':
            probs = np.array([e.state.sum() for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
                
        elif mode == 'reward':
            probs = np.array([e.reward + 1 for e in self.memory]) 
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
                
        elif mode == 'error':
            probs = np.array([e.error for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'error_u':
            probs = np.array([e.error for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'weighted_error':
            weights = np.array([e.weight for e in self.memory])
            max_weight = weights.max()
            sum_weight = weights.sum()
            weights = np.array([(max_weight - w + 1) / (sum_weight + len(weights)) for w in weights])
            probs = np.array([e.error for e in self.memory])
            probs = probs * weights
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'weighted_error_reversed':
            weights = np.array([e.weight for e in self.memory])
            sum_weight = weights.sum()
            weights = np.array([(w) / (sum_weight) for w in weights])
            probs = np.array([e.error for e in self.memory])
            probs = probs * weights
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'action_balanced_error':
            probs = np.array([e.error * e.action_dist for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'clipped_error':
            t = pd.DataFrame(self.memory)
            
            t = t[t['error'] < t['error'].quantile(0.99)]
            t['probs'] = t['error'] * t['action_dist']
            t['probs'] = t['probs'] / t['probs'].sum()
            idx = np.random.choice(len(t), size=self.batch_size, p=t['probs'].values)
            t = t.iloc[idx]
            
            experiences = deque(maxlen=self.batch_size)        
            for i in list(t.itertuples(name='Experience', index=False)):
                experiences.append(i)
                
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)