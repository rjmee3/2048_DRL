import numpy as np
import matplotlib.pyplot as plt
import random

'''--------- Move Constants --------'''
ACTION_UP = 0
ACTION_DN = 1
ACTION_LT = 2
ACTION_RT = 3

'''--------- Base Directory --------'''
base_dir = '.'

'''-------- 2048 Game Class --------'''
class Game():
    def __init__(self, size=4, seed=42, negative_reward=-2, tile_move_penalty=0.1):
        '''
        Initializes the game environment. size parameter is used in
        setting the dimensions of the board. seed is used in seeding
        random generation. negative_reward determines how penalized
        an invalid move is. reward_mode determines the reward scheme.
        tile_move_penalty is the penalty to moving an individual tile. '''
        
        self.board_dim = size
        self.state_size = size * size
        self.action_size = 4
        self.best_game_history = []
        self.negative_reward = negative_reward
        self.tile_move_penalty = tile_move_penalty
        
        np.random.seed(seed)
            
    def reset(self, start_tiles_num=2):
        ''' 
        Resets board and all variables used in tracking a single game. 
        start_tiles_num parameter is used to determine number of tiles 
        to spawn on the initial board.'''
        
        # creating empty board
        self.game_board = np.zeros((self.board_dim, self.board_dim))
        
        # spawn number of tiles specified by start_tiles_num
        for _ in range(start_tiles_num):
            self.generate_random_tile()
            
        # resetting everything on per game basis
        self.score = np.sum(self.game_board)
        self.reward = 0
        self.current_cell_move_penalty = 0
        self.done = False
        self.steps = 0
        self.rewards_list = []
        self.scores_list = []
        self.history = []
        
        # append new state to history
        self.history.append({
            'action': -1,
            'new_board': self.game_board.copy(),
            'old_board': None,
            'score': self.score,
            'reward': self.reward
        })
        
    def shift(self, board):
        '''
        Shifts all tiles to the left. Calculates cell moving penalty.
        Does not merge tiles. '''
        
        # creating an empty board of same size as original
        shifted_board = np.zeros_like(board)
        
        # shift all tiles left
        for i, row in enumerate(board):
            shifted = np.zeros(len(row))
            index = 0
            for j, val in enumerate(row):
                if val != 0:
                    shifted[index] = val
                    
                    # increasing move penalty if tiles move
                    if j != index:
                        self.current_cell_move_penalty += self.tile_move_penalty * val
                        
                    index += 1
            shifted_board[i] = shifted
            
        return shifted_board
    
    def calc_board(self, board):
        '''
        Handles tile merging and reward calculation. '''
        
        self.reward = 0
        self.current_cell_move_penalty = 0
        
        # shift all tiles left
        shifted_board = self.shift(board)
        
        # merging tiles
        merged_board = np.zeros_like(shifted_board)
        for index, row in enumerate(shifted_board):
            for i in range(len(row)-1):
                if row[i] != 0 and row[i] == row[i+1]:
                    row[i] = row[i] * 2
                    row[i+1] = 0
                    
                    # calculating reward
                    self.reward += np.log2(row[i])
                        
            merged_board[index] = row
            
        # after merging tiles, shift tiles left again
        merged_board = self.shift(merged_board)
                
        return merged_board
    
    
    def calculate_merges_value(self, board):
        '''
        Used in reward calculation to encourage moves which create
        merge opportunities by placing like tiles next to each other. 
        NOT CURRENTLY IN USE. '''
        merge_values = 0
        
        # for horizontal merges
        for row in range(self.board_dim):
            squeezed_row = np.trim_zeros(board[row, :], 'b')
            for col in range(len(squeezed_row) - 1):
                if squeezed_row[col] == squeezed_row[col + 1] and squeezed_row[col] != 0:
                    if self.reward_mode == 'log2':
                        merge_values += np.log2(squeezed_row[col])
                    else:
                        merge_values += 2 * squeezed_row[col]
                    col += 1
                    
        # for vertical merges
        for col in range(self.board_dim):
            squeezed_col = np.trim_zeros(board[:, col], 'b')
            for row in range(len(squeezed_col) - 1):
                if squeezed_col[row] == squeezed_col[row + 1] and squeezed_col[row] != 0:
                    if self.reward_mode == 'log2':
                        merge_values += np.log2(squeezed_col[row])
                    else:
                        merge_values += 2 * squeezed_col[row]
                    row += 1
                    
        return merge_values
    
    def current_state(self):
        '''
        Returns flattened current board. '''
        return np.reshape(self.game_board.copy(), -1)
    
    def step(self, action, action_values):
        '''
        Takes a step based on the action passed. '''
        
        # making copies of the board to use in moving and history saving
        old_board = self.game_board.copy()
        temp_board = self.game_board.copy()
        
        # move based on action passed
        if action == ACTION_LT:
            temp_board = self.calc_board(temp_board)
        elif action == ACTION_RT:
            temp_board = np.flip(
                self.calc_board(
                    np.flip(temp_board, axis=1)), axis=1)
        elif action == ACTION_UP:
            temp_board = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(temp_board), axis=0)), axis=0))
        elif action == ACTION_DN:
            temp_board = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(temp_board), axis=1)), axis=1))
        else:
            return (self.game_board, 0, self.done)
        
        # handling valid and invalid moves
        if not np.array_equal(self.game_board, temp_board):
            # operations for a valid move
            self.game_board = temp_board.copy()
            self.generate_random_tile()
            self.reward = self.reward - self.current_cell_move_penalty
            self.score = np.sum(self.game_board)
            self.done = self.check_is_done()
            self.moved = True
        else:
            #operations for an invalid move
            self.reward = self.negative_reward
            self.moved = False
        self.steps += 1
        self.rewards_list.append(self.reward)
        
        # append move to history
        self.history.append({
            'action': action,
            'action_values': action_values,
            'new_board': self.game_board.copy(),
            'old_board': old_board,
            'score': self.score,
            'reward': self.reward
        })
        
        # returning resulting board, reward, and done condition
        return (self.game_board, self.reward, self.done)
    
    def check_is_done(self, board=None):
        '''
        Used in updating done condition. '''
        if board is None:
            board = self.game_board.copy()
            
        if not np.all(board):
            return False
        else:
            for row in board:
                for cell in range(len(row)-1):
                    if row[cell] == row[cell+1]:
                        return False
            for row in np.transpose(board):
                for cell in range(len(row)-1):
                    if row[cell] == row[cell+1]:
                        return False
                    
            return True
    
    def generate_random_tile(self):
        '''
        Used in generating a random tile on the board. '''
        
        if np.all(self.game_board):
            return
        
        empty_cells = np.argwhere(self.game_board == 0)
        i, j = random.choice(empty_cells)
        self.game_board[i, j] = np.random.choice([2, 4], p=[0.9, 0.1])

    def draw_board(self, board=None, title='Current Game'):
        if board is None:
            board = self.game_board
        num_cols = self.board_dim
        num_rows = self.board_dim
        fig = plt.figure(figsize=(3, 3))
        plt.suptitle(title)
        axes = [fig.add_subplot(num_rows, num_cols, r * num_cols + c) for r in range(num_rows) for c in range(1, num_cols + 1)]
        v = np.reshape(board, -1)
        cell_colors = {
            0: '#FFFFFF', 2: '#EEE4DA', 4: '#ECE0C8', 8: '#ECB280', 16:'#EC8D53', 32:'#F57C5F', 64:'#E95937',
            128:'#F3D96B', 256:'#F2D04A', 512:'#E5BF2E', 1024:'#E2B814', 2048:'#EBC502', 4096:'#00A2D8',
            8192:'#9ED682', 16384:'#9ED682', 32768:'#9ED682', 65536:'#9ED682', 131072:'#9ED682'
        }
        for i, ax in enumerate(axes):
            ax.text(0.5, 0.5, str(int(v[i])), horizontalalignment='center', verticalalignment='center')
            ax.set_facecolor(cell_colors[int(v[i])])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()