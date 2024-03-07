import torch
import torch.nn as nn
import torch.optim as optim
from model import QNetwork
from state import State
import numpy as np
import pandas as pd
import os

'''----------------------------HYPERPARAMETERS----------------------------'''

BOARD_SIZE         = 4                                              # length of one size of the board
BATCH_SIZE         = 1                                              # num of examples used in one iteration
STATE_SIZE         = BATCH_SIZE * (BOARD_SIZE**4 + BOARD_SIZE**2)   # size of a one-hot encoded board
ACTION_SIZE        = 4                                              # will always be 4 (up, down, left, right)
LEARNING_RATE      = 0.001
GAMMA              = 0.99
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10
MAX_EPISODES       = 10000
EPSILON_INITIAL    = 0.1
EPSILON_DECAY      = 0.99
MODEL_WEIGHTS_PATH = 'model_weights.pth'

'''-----------------------------------------------------------------------'''


# creating data frame to store episode info
episode_data = {'Episode': [], 'Total Reward': [], 'Move Count': [], 'Up Count': [], 'Down Count': [], 
                'Left Count': [], 'Right Count': [], 'Invalid Count': [], 'Max Tile': []}

# instantiating networks
model = QNetwork(STATE_SIZE, ACTION_SIZE)
target_model = QNetwork(STATE_SIZE, ACTION_SIZE)

# loading model weights if possible
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

#training loop
for episode in range(MAX_EPISODES):
    # reset the board each episode
    state.reset()
    
    # reward per episode
    total_reward = 0
    
    # reseting epsilon
    epsilon = EPSILON_INITIAL
    
    while not state.game_over:
        # convert state to one-hot encoded tensor
        state_tensor = state.one_hot_encode(BATCH_SIZE)
        
        # choosing action using epsilon-greedy strat
        valid_actions = state.get_valid_actions()
        invalid_actions = np.array(state.get_invalid_actions())
        if np.random.rand() < epsilon:
            action = np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                
                if len(invalid_actions) > 0:
                    q_values[0, invalid_actions] = float('-inf')
                action = torch.argmax(q_values).item()
                
        # applying selected action
        state.move(action)
        
        # calculate reward
        reward = state.calculate_reward()
        
        #increment total reward
        total_reward += reward
        
        # update game_over
        state.update()
        
        # decay epsilon
        epsilon *= EPSILON_DECAY
        
        # get next state as one-hot encoded tensor
        next_state_tensor = state.one_hot_encode(BATCH_SIZE)
        
        # store experience in replay buffer
        replay_buffer.append((state_tensor, action, reward, next_state_tensor, state.game_over))
        
        # sampling a mini-batch from replay buffer
        if len(replay_buffer) >= BATCH_SIZE:
            indices = torch.randint(len(replay_buffer), size=(BATCH_SIZE,), dtype=torch.long)
            mini_batch = [replay_buffer[i] for i in indices]
            
            # extracting components
            states, actions, rewards, next_states, dones = zip(*mini_batch)
            
            # converting to pytorch tensors
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            # calculating target q vals
            with torch.no_grad():
                target_q_values = target_model(next_states)
                target_q_values, _ = torch.max(target_q_values, dim=1)
                target_q_values = rewards + GAMMA * (1 - dones) * target_q_values
                
            # concatenate the tensors in the tuple along dimension 0
            states_concatenated = torch.cat(states, dim=0)

            # passing the concatenated tensor to the model
            predicted_q_values = model(states_concatenated).gather(1, actions.view(-1, 1))
            
            # compute loss and update model
            loss = criterion(predicted_q_values, target_q_values.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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
    
df = pd.DataFrame(episode_data)
df.to_excel('episode_data.xlsx', index=False)

torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)