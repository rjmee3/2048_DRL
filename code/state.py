from collections import deque
import numpy as np
import random
import torch
from copy import copy

'''----------------------------HYPERPARAMETERS----------------------------'''
EMPTY_WEIGHT  = 100
MONO_WEIGHT   = 10
SMOOTH_WEIGHT = 1

# function to place a random tile on the board
def place_rand_tile(board):
    # creating a list of each tile coord with a zero
    empty_cells = np.argwhere(board == 0)
    
    # if any empty cells exist, randomly choose one and place 
    # either a 2 (90% chance) or a 4 (10% chance)
    if len(empty_cells) > 0:
        i, j = random.choice(empty_cells)
        board[i, j] = 2 if random.random() < 0.9 else 4

# function to merge board matrix.
def merge(board, score):
    queue = deque()
        
    # queuing all non-zero elements in a row
    for row in board:
        for i in range(len(row)):
            if row[i] != 0:
                queue.append(row[i])
                
        index = 0
        
        # deque elements, merging like elements
        while queue:
            row[index] = queue.popleft()
            
            if queue and row[index] == queue[0]:
                row[index] += queue.popleft()
                score += row[index]
            
            index += 1
            
        # set all further elements to zero
        while index < len(row):
            row[index] = 0
            index += 1
            
    return score

class State: 
    
    '''
                    BOARD MANIPULATION FUNCTIONS        
            --------------------------------------------
        Functions related to manipulating the data on the board.
        
        __init__         :  creates a BOARD_SIZE x BOARD_SIZE array to 
                            represent the board, and then places 2 random
                            tiles on the board. Also initializes several
                            parameters to zero, and initializes game_over
                            to false.
                    
        reset            :  Basically does exactly what __init__ does, just
                            resets an already initialized object.
                    
        move             :  Applies the move provided by the direction
                            parameter, also increments the corresponding
                            move counter and score.
                    
        update           :  Checks the board for game over and updates the
                            game_over attribute if there are no moves left
                            to make.
    '''
    
    def __init__(self, board_size):
        self.board = np.zeros((board_size, board_size), dtype=int)
        place_rand_tile(self.board)
        place_rand_tile(self.board)
        self.board_size = board_size
        self.game_over = False
        self.move_count = 0
        self.invalid_count = 0
        self.up_count = 0
        self.down_count = 0
        self.left_count = 0
        self.right_count = 0
        self.score = 0
        self.prev_score = 0
        self.max = 0
        self.prev_max = 0
        self.value = self.calculate_board_value()
        self.prev_value = self.value
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        place_rand_tile(self.board)
        place_rand_tile(self.board)
        self.game_over = False
        self.move_count = 0
        self.invalid_count = 0
        self.up_count = 0
        self.down_count = 0
        self.left_count = 0
        self.right_count = 0
        self.score = 0
        self.prev_score = 0
        self.max = 0
        self.prev_max = 0
        self.value = self.calculate_board_value()
        self.prev_value = self.value
            
    def move(self, direction, count=True):
        orig_board = np.copy(self.board)
        self.prev_score = self.score
        self.prev_max = np.max(self.board)
        self.prev_value = self.value
        if direction == 0:    # LEFT
            self.score = merge(self.board, self.score)
            if count:
                self.left_count += 1
            pass
        elif direction == 1:  # RIGHT
            self.board = np.flip(self.board, axis=1)
            self.score = merge(self.board, self.score)
            self.board = np.flip(self.board, axis=1)
            if count:
                self.right_count += 1
            pass
        elif direction == 2:  # UP
            self.board = np.transpose(self.board)
            self.score = merge(self.board, self.score)
            self.board = np.transpose(self.board)
            if count:
                self.up_count += 1
            pass
        elif direction == 3:  # DOWN
            self.board = np.transpose(self.board)
            self.board = np.flip(self.board, axis=1)
            self.score = merge(self.board, self.score)
            self.board = np.flip(self.board, axis=1)
            self.board = np.transpose(self.board)
            if count:
                self.down_count += 1
            pass
        else:
            print("ERR: Invalid Action.")
            
        if not np.array_equal(self.board, orig_board):
            place_rand_tile(self.board)
            self.max = np.max(self.board)
            self.value = self.calculate_board_value()
        else:
            self.invalid_count += 1
            
        if count:
                self.move_count += 1

    # function updates game over attribute
    def update(self):
        if np.any(self.board == 0):
            self.game_over = False
            return
        
        if np.any(np.diff(self.board, axis=1) == 0) or np.any(np.diff(self.board, axis=0) == 0):
            self.game_over = False
            return
            
        self.game_over = True
        
    
    '''
                     DATA ACCESSING FUNCTIONS        
            --------------------------------------------
        Functions related to getting data from the state object.
        
        print            :  prints the current state of the board.
    '''
    
    def print(self):
        for row in self.board:
            format_row = ' '.join(f'{value:5}' for value in row)
            print(format_row, flush=True)  
        
    '''
                    MACHINE LEARNING FUNCTIONS        
            --------------------------------------------
        Functions related to using the state in a machine learning
        environment.
        
        one_hot_encode   :  Returns a one-hot encoded tensor of size 
                            1 x BOARD_SIZE x BOARD_SIZE x (BOARD_SIZE**2)+1 
                            (+1 because that accounts for the highest 
                            attainable tile from randomly getting a 4)
        
        calculate_reward :  Returns the reward from making a move, which 
                            is simply calculated as the current score - 
                            previous score. (Points are earned towards 
                            score by merging tiles, with the score earned
                            being the value of the merged tile.)
    '''
    
    def one_hot_encode(self):
        log_values = np.log2(self.board[self.board != 0]).astype(np.float32)
        
        one_hot_encoded = np.zeros((self.board_size, self.board_size, (self.board_size**2)+1), dtype=np.int32)
        
        for i in range(len(log_values)):
            row, col = np.where(self.board == 2**log_values[i])
            one_hot_encoded[row, col, int(log_values[i])] = 1
            
        return torch.flatten(torch.from_numpy(one_hot_encoded).to(torch.float), start_dim=0, end_dim=1)
    
    def calculate_reward(self):
        return (self.score - self.prev_score) + self.calculate_merges_value()
    
            # (2 * (self.max - self.prev_max))
            # (self.score - self.prev_score)    + )    # increase in score
                #      # value of merges on the board
                  # increase in max tile
                # + (self.value - self.prev_value))   # increase in inherent "board value"
        
    
    def get_valid_actions(self):
        valid_actions = []
        
        orig_board = np.copy(self.board)
        orig_score = self.score
        orig_prev_score = self.prev_score
        orig_invalid_count = self.invalid_count
        
        for action in range(4):
            self.move(action, count=False)
            if self.invalid_count == orig_invalid_count:
                valid_actions.append(action)
            
            self.board = np.copy(orig_board)
            self.score = orig_score
            self.prev_score = orig_prev_score
            self.invalid_count = orig_invalid_count
        
        return valid_actions
    
    def get_invalid_actions(self):
        invalid_actions = []
        
        orig_board = np.copy(self.board)
        orig_score = self.score
        orig_prev_score = self.prev_score
        orig_invalid_count = self.invalid_count
        
        for action in range(4):
            self.move(action, count=False)
            if self.invalid_count != orig_invalid_count:
                invalid_actions.append(action)
            
            self.board = np.copy(orig_board)
            self.score = orig_score
            self.prev_score = orig_prev_score
            self.invalid_count = orig_invalid_count
        
        return invalid_actions
    
    def calculate_merges_value(self):
        merge_values = 0
        
        # for horizontal merges
        for row in range(self.board_size):
            squeezed_row = np.trim_zeros(self.board[row, :], 'b')
            for col in range(len(squeezed_row) - 1):
                if squeezed_row[col] == squeezed_row[col + 1] and squeezed_row[col] != 0:
                    merge_values += 2 * squeezed_row[col]
                    
        # for vertical merges
        for col in range(self.board_size):
            squeezed_col = np.trim_zeros(self.board[:, col], 'b')
            for row in range(len(squeezed_col) - 1):
                if squeezed_col[row] == squeezed_col[row + 1] and squeezed_col[row] != 0:
                    merge_values += 2 * squeezed_col[row]
                    
        return merge_values
    
    def calculate_board_value(self):
        # get number of empty cells
        board_1d = self.board.flatten()
        empty_cells = np.count_nonzero(board_1d == 0)
        
        # calculate monotonicity
        row_monotonicity = np.sum(np.diff(np.sign(np.diff(self.board, axis=1)), axis=1) >= 0)
        col_monotonicity = np.sum(np.diff(np.sign(np.diff(self.board, axis=0)), axis=0) >= 0)
        monotonicity = max(row_monotonicity, col_monotonicity)
        
        # calculate smoothness
        smoothness = 0
        for row in self.board:
            smoothness -= np.sum(np.abs(np.diff(row)))
        for col in self.board.T:
            smoothness -= np.sum(np.abs(np.diff(col)))
            
        # compute inherent value based on these metrics
        value = empty_cells * EMPTY_WEIGHT + monotonicity * MONO_WEIGHT + smoothness * SMOOTH_WEIGHT
        
        return value