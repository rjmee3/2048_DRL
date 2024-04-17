import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        
        self.episode_memory = []
        self.batch_size = batch_size
        
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "error"])
        self.geomspaces = [np.geomspace(1., 0.5, i) for i in range(1, 10)]
            
    def reset_episode_memory(self):
        self.episode_memory = []
            
    def add(self, state, action, reward, next_state, done, error):
        e = self.experience(state, action, reward, next_state, done, error)
        self.episode_memory.append(e)
        
    def add_episode_experience(self):
        self.memory.extend(self.episode_memory)
        self.reset_episode_memory()
        
    def calc_expected_rewards(self, steps_ahead=1):
        rewards = [e.reward for e in self.episode_memory if e is not None]
        
        exp_rewards = [np.sum(rewards[i:i+steps_ahead] * self.geomspaces[steps_ahead-1]) for i in range(len(rewards)-steps_ahead)]

        temp_memory = []
        
        for i, e in enumerate(self.episode_memory[:-steps_ahead]):
            t_e = self.experience(e.state, e.action, exp_rewards[i], e.next_state, e.done, e.error)
            temp_memory.append(t_e)
            
        self.episode_memory = temp_memory
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
                
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)