import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from model import QNetwork
from replay_buffer import ReplayBuffer
import time
from IPython.display import clear_output

# Constants
BUFFER_SIZE = 100000
BATCH_SIZE = 1024
LR = 0.00005

# Base directory for saving/loading model data
base_dir = './data/'

# Setting up device for Torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transform_state(state):
    '''
    Reshapes state into a one-hot encoded vector. 
    Copied from training_functions to prevent circular dependency. '''
    state = np.reshape(state, -1)
    state[state==0] = 1
    state = np.log2(state)
    state = state.astype(int)
    new_state = np.reshape(np.eye(18)[state], -1)
    return new_state

class Agent:
    def __init__(self, state_size=16, action_size=4, seed=42,
                 fc1_units=256, fc2_units=256, fc3_units=256,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 lr=LR, use_expected_rewards=True, predict_steps=2):
        ''' 
        Initializes agent based on passed args. Creates neural network and
        data collection lists/objects for training metric tracking. '''

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.batch_size = batch_size
        self.losses = []
        self.use_expected_rewards = use_expected_rewards
        self.current_iteration = 0

        # Data collections for plotting and analysis
        self.last_n_scores = deque(maxlen=50)
        self.max_score = 0
        self.best_score_board = []
        self.last_n_total_rewards = deque(maxlen=50)
        self.mean_total_rewards = []
        self.max_vals_list = []
        self.best_val_board = []
        self.max_val = 0
        self.max_steps_list = []
        self.last_n_steps = deque(maxlen=50)
        self.mean_steps = []
        self.max_steps = 0
        self.total_steps = 0
        self.best_steps_board = []

        # Initialize Q-Networks and Optimizer
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units, fc2_units, fc3_units).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units, fc2_units, fc3_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9999)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.t_step = 0
        self.steps_ahead = predict_steps
        
    def save(self, name):
        '''
        Save the current state of the network to external files. '''
        torch.save(self.qnetwork_local.state_dict(), f'{base_dir}/network_local_{name}.pth')
        torch.save(self.qnetwork_target.state_dict(), f'{base_dir}/network_target_{name}.pth')
        torch.save(self.optimizer.state_dict(), f'{base_dir}/optimizer_{name}.pth')
        torch.save(self.lr_decay.state_dict(), f'{base_dir}/lr_schd_{name}.pth')

    def load(self, name):
        '''
        Load the model, optimizer, and scheduler states from files. '''
        self.qnetwork_local.load_state_dict(torch.load(f'{base_dir}/network_local_{name}.pth'))
        self.qnetwork_target.load_state_dict(torch.load(f'{base_dir}/network_target_{name}.pth'))
        self.optimizer.load_state_dict(torch.load(f'{base_dir}/optimizer_{name}.pth'))
        self.lr_decay.load_state_dict(torch.load(f'{base_dir}/lr_schd_{name}.pth'))

    def step(self, state, action, reward, next_state, done, error):
        '''
        Save step data in replay buffer. '''
        self.memory.add(state, action, reward, next_state, done, error)

    def act(self, state):
        '''
        Returns actions for given state as per current policy. '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        return action_values.cpu().data.numpy()

    def learn(self, learn_iterations, save_loss=True):
        '''
        Update policy and value parameters using given batch of experience tuples. '''
        if self.use_expected_rewards:
            self.memory.calc_expected_rewards(self.steps_ahead)

        self.memory.add_episode_experience()

        if len(self.memory) > self.batch_size:
            losses = []
            for _ in range(learn_iterations):
                states, actions, rewards, _, _ = self.memory.sample()
                Q_expected = self.qnetwork_local(states).gather(1, actions)
                loss = F.mse_loss(Q_expected, rewards)
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.lr_decay.step()

            if save_loss:
                self.losses.append(np.mean(losses))
        else:
            self.losses.append(0)

    def play(self, env, num_games=1, animate=False, frame_delay=0.25):
        ''' 
        Lets the agent play a set number of games, solely off neural network policy. '''
        for i in range(num_games):
            env.reset(2)
            state = transform_state(env.current_state())

            while not env.done:
                action_values = self.act(state)
                actions_sorted = [(index, value) for index, value in enumerate(action_values[0])]
                actions_sorted = sorted(actions_sorted, key=lambda x: x[1], reverse=True)
                action_index = 0
                env.moved = False

                while not env.moved:
                    action_elem = actions_sorted[action_index]
                    action_index += 1

                    action = np.int64(action_elem[0])

                    env.step(action, action_values)
                    next_state = transform_state(env.current_state())

                    state = next_state

                    if animate:
                        time.sleep(frame_delay)
                        clear_output()


                    env.draw_board(title=f"Game {i+1}")