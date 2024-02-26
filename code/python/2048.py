#!/usr/bin/env python3
from tensorflow import keras 
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from collections import deque
import copy
import random
import os
import pandas as pd
import sys

# program constants
BOARD_SIZE = 4
NEG_REWARD_RATE = -1000
POS_REWARD_RATE = 100

# global variable to track cumulative reward
cumulative_reward = 0

episode_data = pd.DataFrame(columns=['Episode', 'Total Reward', 'Max Tile'])

# function to print the board to the console
def print_board(board):
    # clear the console prior to displaying board
    # os.system('clear' if os.name == 'posix' else 'cls')
    for i in range(0, len(board), BOARD_SIZE):
        format_row = ' '.join(f'{value:5}' for value in board[i:i+BOARD_SIZE])
        print(format_row)
        sys.stdout.flush()
        
# function to initialize the board 
def initialize_board():
    board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    place_rand_tile(board)
    place_rand_tile(board)
    return flatten_board(board)

# function which returns board as a flattened vector
def flatten_board(board):
    return [value for row in board for value in row]

# function to reshape board back to a BOARD_SIZE x BOARD_SIZE array
def reshape_board(board):
    return [board[i:i+BOARD_SIZE] for i in range(0, len(board), BOARD_SIZE)]

# function to place a random tile on the board
def place_rand_tile(board):
    # creating a list of each tile coord with a zero
    empty_cells = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]
    
    # if any empty cells exist, randomly choose one and place 
    # either a 2 (90% chance) or a 4 (10% chance)
    if empty_cells:
        i, j = random.choice(empty_cells)
        board[i][j] = 2 if random.random() < 0.9 else 4
      
# function to handle moves in all directions
def move(board, action):
    global cumulative_reward
    board = reshape_board(board)
    orig_board = copy.deepcopy(board)
    # LEFT
    if action == 0:
        board = merge(board)
        pass
    # RIGHT
    elif action == 1:
        board = reverse(board)
        board = merge(board)
        board = reverse(board)
        pass
    # UP
    elif action == 2:
        board = transpose(board)
        board = merge(board)
        board = transpose(board)
        pass
    # DOWN
    elif action == 3:
        board = transpose(board)
        board = reverse(board)
        board = merge(board)
        board = reverse(board)
        board = transpose(board)
        pass
    else:
        print("ERR: Invalid Action. ")
        return flatten_board(board), calculate_reward(orig_board, board)
    if board != orig_board:
        place_rand_tile(board)
        
    reward = calculate_reward(orig_board, board)
    cumulative_reward += reward
    return flatten_board(board), reward

def calculate_reward(orig_board, board):
    reward = 0
    board = flatten_board(board)
    orig_board = flatten_board(orig_board)
    if board == orig_board:
        reward += NEG_REWARD_RATE

    # reward += (max(board) - max(orig_board)) * POS_REWARD_RATE * 2
    
    reward += (board.count(0) - orig_board.count(0)) * POS_REWARD_RATE
    return reward

# function to transpose board matrix.
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

# function to reverse board matrix.
def reverse(matrix):
    return [row[::-1] for row in matrix]

# funcytion to merge board matrix.
def merge(board):
    queue = deque()
    
    for row in board:
        for i in range(BOARD_SIZE):
            if row[i] != 0:
                queue.append(row[i])
                
        index = 0
        while queue:
            row[index] = queue.popleft()
            
            if queue and row[index] == queue[0]:
                row[index] += queue.popleft()
            
            index += 1
            
        while index < BOARD_SIZE:
            row[index] = 0
            index += 1

    return board

# function to check if the game is over
def is_game_over(board):
    board = reshape_board(board)
    for row in board:
        if 0 in row or any(row[i] == row[i+1] for i in range(BOARD_SIZE-1)):
            return False
        
    for col in range(len(board[0])):
        col_val = [board[row][col] for row in range(BOARD_SIZE)]
        if 0 in col_val or any(col_val[i] == col_val[i+1] for i in range(BOARD_SIZE-1)):
            return False
        
    return True

# defining neural network architecture
def build_model(input_size, output_size):
    model = Sequential([
        Dense(64, input_dim=input_size, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_size, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')
    return model

def one_hot_encode(board):
    possible_values = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    encoded_board = np.zeros((4, 4, 16), dtype=int)
    
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            tile_value = board[i][j]
            if tile_value in possible_values:
                encoded_board[i, j, possible_values.index(tile_value)] = 1
                
    return encoded_board

if __name__ == "__main__":
    
    # board = initialize_board()
    # while not is_game_over(board):
    #     print_board(board)
    #     direction = input("Enter Move: ")
    #     board = move(board, direction)
        
    # print_board(board)
    # print("Game Over!")
    
    # building the policy network
    policy_model = build_model((BOARD_SIZE * BOARD_SIZE) ** 2, 4)
    
    # hyperparameters
    num_episodes = 1000
    gamma = 0.99
    epsilon = 0.1
    episode_states = []
    episode_actions = []
    episode_rewards = []
    
    # training loop
    for episode in range(num_episodes):
        board = initialize_board()  # Initialize the game board for a new episode
        total_reward = 0  # Accumulate the total reward for this episode
        state_sequence = []  # Store the states for this episode
        action_sequence = []  # Store the chosen actions for this episode
        reward_sequence = []  # Store the rewards for this episode
        epsilon *= 0.995  # Annealing epsilon over time

        while not is_game_over(board):
            state = np.array([one_hot_encode(reshape_board(board)).reshape(-1)])  # Convert the current board to a flattened vector

            # Choose an action based on the policy probabilities
            action_probabilities = policy_model.predict(state, verbose=0)[0]
            # Add exploration-exploitation strategy (epsilon-greedy)
            action = np.argmax(action_probabilities) if np.random.rand() < epsilon else np.random.choice(4)

            # Take the action and observe the new state and reward
            result = move(board, action)
            new_board, reward = result[0], result[1]

            # Store the state, action, and reward
            state_sequence.append(one_hot_encode(reshape_board(new_board)).reshape(-1))
            action_sequence.append(action)
            reward_sequence.append(reward)

            total_reward += reward
            board = new_board            

        # Calculate discounted returns
        discounted_returns = []
        cumulative_return = 0
        for t in reversed(range(len(reward_sequence))):
            cumulative_return = cumulative_return * gamma + reward_sequence[t]
            discounted_returns.insert(0, cumulative_return)

        # Convert actions to one-hot encoded vectors
        actions_one_hot = np.eye(4)[action_sequence]

        # Convert lists to numpy arrays for training
        states_array = np.vstack(state_sequence)
        returns_array = np.vstack(discounted_returns)

        # Train the policy model using policy gradient update
        policy_model.train_on_batch(states_array, actions_one_hot, sample_weight=returns_array)

        # Print episode information
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        print_board(board)
        
        # appending data to data frame
        episode_data = episode_data._append({'Episode': episode + 1, 'Total Reward': total_reward, 'Max Tile': max(board)}, ignore_index=True)

    # Save the trained policy model if needed
    policy_model.save('policy_model.h5')
    
    # save data to excel file
    episode_data.to_excel('episode_data.xlsx', index=False)
    