import torch
import torch.nn as nn
import torch.optim as optim
from model import QNetwork
from state import State
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# custom dataset class
class ReplayBufferDataset(Dataset):
    def __init__(self, replay_buffer, buffer_size):
        self.replay_buffer = replay_buffer
        self.buffer_size = buffer_size

    def __len__(self):
        return min(len(self.replay_buffer), self.buffer_size)

    def __getitem__(self, idx):
        return self.replay_buffer[idx]
    
# function for softmax exploration
def softmax(x):
    """Compute softmax values for each row of input x."""
    e_x = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
    return e_x / e_x.sum(axis=0)

'''----------------------------HYPERPARAMETERS----------------------------'''

BOARD_SIZE         = 4                                              # length of one size of the board
BATCH_SIZE         = 100                                            # num of examples used in one iteration
STATE_SIZE         = (BOARD_SIZE**4 + BOARD_SIZE**2)                # size of a one-hot encoded board
ACTION_SIZE        = 4                                              # will always be 4 (up, down, left, right)
LEARNING_RATE      = 0.001                                          # learning rate for minimizing loss function
GAMMA              = 0.9                                            # discount factor to determine importance of future reward
REPLAY_BUFFER_SIZE = 10000                                          # how many experiences are stored in the replay buffer
TARGET_UPDATE_FREQ = 1                                              # how many episodes it takes until the target network is updated
MAX_EPISODES       = 10000                                          # maximum number of episodes to train on
EPSILON_INITIAL    = 1.0                                            # initial value for epsilon in epsilon-greedy strategy
EPSILON_DECAY      = 0.85                                           # the rate at which epsilon decays
EPSILON_FINAL      = 0.01                                           # the lowest epsilon is allowed to go
MODEL_WEIGHTS_PATH = 'model_weights.pth'                            # file path for saved model weights

'''-----------------------------------------------------------------------'''


# creating data frame to store episode info
episode_data = {'Episode': [], 'Total Reward': [], 'Move Count': [], 'Up Count': [], 'Down Count': [], 
                'Left Count': [], 'Right Count': [], 'Invalid Count': [], 'Max Tile': []}

# instantiating networks
model = QNetwork(STATE_SIZE, ACTION_SIZE)
target_model = QNetwork(STATE_SIZE, ACTION_SIZE)

# loading model weights if possible/available
if os.path.exists(MODEL_WEIGHTS_PATH):
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
target_model.load_state_dict(model.state_dict())

# defining loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# initializing replay buffer
replay_buffer = []

# initializing state object
state = State(BOARD_SIZE)

# establish matplotlib for interactive mode
plt.ion()
fig, ax = plt.subplots(3, 1, figsize=(12, 9))

# Plot initial data
reward_plot, = ax[0].plot([], [], label='Total Reward', color='blue')
ax[0].set_title('Total Reward per Episode')
ax[0].set_xlabel('Episode')
ax[0].set_ylabel('Total Reward')
ax[0].legend()

loss_plot, = ax[1].plot([], [], label='Total Loss', color='blue')
ax[1].set_title('Loss over Each Episode')
ax[1].set_xlabel('Training Step')
ax[1].set_ylabel('Loss')
ax[1].legend()

scatter_plot, = ax[2].plot([], [], label='Total Reward vs Total Moves', color='red', marker='o', linestyle='')
ax[2].set_title('Moving Average of Total Reward')
ax[2].set_xlabel('# of Moves')
ax[2].set_ylabel('Total Reward')
ax[2].legend()

# initialize empty lists for dynamic plotting
episode_list = []
step_list = []
reward_list = []
loss_list = []
move_list = []

#training loop
for episode in range(MAX_EPISODES):
    # reset everything
    state.reset()
    total_reward = 0
    epsilon = EPSILON_INITIAL
    step_count = 0
    
    # allow network to play full game
    while not state.game_over:
        # incrementing step count
        step_count += 1
        
        # convert state to one-hot encoded tensor
        state_tensor = state.one_hot_encode()
        
        # choosing action using softmax exploration
        invalid_actions = np.array(state.get_invalid_actions())
        with torch.no_grad():
            q_values = model(state_tensor)
            if len(invalid_actions) > 0:
                q_values[0, invalid_actions] = float('-inf')
        q_values_numpy = q_values.cpu().numpy().flatten()
        action_probabilities = softmax(q_values_numpy)
        action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
        state.move(action)
        
        # epsilon greedy approach
        # valid_actions = np.array(state.get_valid_actions())
        # invalid_actions = np.array(state.get_invalid_actions())
        # if np.random.rand() < epsilon:
        #     action = np.random.choice(valid_actions)
        # else:
        #     with torch.no_grad():
        #         q_values = model(state_tensor)
        #         if len(invalid_actions) > 0:
        #             q_values[0, invalid_actions] = float('-inf')
        #         action = torch.argmax(q_values).item()
        # state.move(action)
        
        # calculate reward and increment total reward
        reward = state.calculate_reward()
        total_reward += reward
        
        # update game_over
        state.update()
        
        # decay epsilon
        if epsilon > EPSILON_FINAL:
            epsilon *= EPSILON_DECAY
        
        # get next state as one-hot encoded tensor
        next_state_tensor = state.one_hot_encode()
        
        # store experience in replay buffer
        replay_buffer.append((state_tensor, action, reward, next_state_tensor, state.game_over))
        
        # ensure replay buffer does not exceed maximum size
        if len(replay_buffer) > REPLAY_BUFFER_SIZE:
            replay_buffer = replay_buffer[-REPLAY_BUFFER_SIZE:]
        
        # sampling a mini-batch from replay buffer (FULL BATCH)
        if len(replay_buffer) >= BATCH_SIZE:
            dataset = ReplayBufferDataset(replay_buffer, REPLAY_BUFFER_SIZE)
            dataLoader = DataLoader(dataset, batch_size=len(replay_buffer), shuffle=True)
            
            # for each element in data loader train network?
            # really not sure how this bit works :/
            for batch in dataLoader:
                # extracting components
                states, actions, rewards, next_states, dones = batch
            
                # converting to pytorch tensors
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)
                
                # calculating target q vals
                with torch.no_grad():
                    target_q_values = target_model(next_states)
                    target_q_values, _ = torch.max(target_q_values, dim=1)
                    target_q_values = rewards + GAMMA * (1 - dones) * target_q_values

                # passing the concatenated tensor to the model
                predicted_q_values = model(states).gather(1, actions.view(-1, 1))
                
                # compute loss and update model
                loss = criterion(predicted_q_values, target_q_values.view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # graphing loss each step (might be slowing it down a lot idk)
            step_list.append(step_count)
            loss_list.append(loss.item())
            loss_plot.set_data(step_list, loss_list)
            ax[1].relim()
            ax[1].autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
       
    # reset loss and step list each episode            
    loss_list = []
    step_list = []
            
    # update the target network every TARGET_UPDATE_FREQ episodes
    if episode % TARGET_UPDATE_FREQ == 0:
        target_model.load_state_dict(model.state_dict())
        
    # print the board and its total reward at the end of each episode
    state.print()
    print("Episode", episode, "- Final Reward = ", total_reward, flush=True)
    
    # store episode data
    episode_data['Episode'].append(episode)
    episode_data['Total Reward'].append(total_reward)
    episode_data['Move Count'].append(state.move_count)
    episode_data['Up Count'].append(state.up_count)
    episode_data['Down Count'].append(state.down_count)
    episode_data['Left Count'].append(state.left_count)
    episode_data['Right Count'].append(state.right_count)
    episode_data['Invalid Count'].append(state.invalid_count)
    episode_data['Max Tile'].append(np.max(state.board))
    
    # update plot dynamically
    episode_list.append(episode)
    reward_list.append(total_reward)
    move_list.append(state.move_count)
    reward_plot.set_data(episode_list, reward_list)
    scatter_plot.set_data(move_list, reward_list)
    
    ax[0].relim()
    ax[0].autoscale_view()
    ax[2].relim()
    ax[2].autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
            
plt.ioff()
plt.show()
    
# saving data to excel
df = pd.DataFrame(episode_data)
df.to_excel('episode_data.xlsx', index=False)

# save model weights
torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)