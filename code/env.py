from collections import deque
import numpy as np
import random
import math

# function to place a random tile on the board
def place_rand_tile(board):
    # creating a list of each tile coord with a zero
    empty_cells = np.argwhere(board == 0)
    
    # if any empty cells exist, randomly choose one and place 
    # either a 2 (90% chance) or a 4 (10% chance)
    if len(empty_cells) > 0:
        i, j = empty_cells[0]
        board[i, j] = 2 # if random.random() < 0.9 else 4
        
# function to merge board matrix.
def merge(board):
    queue = deque()
    score = 0
        
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

def move(board, direction):
        score = 0
        invalid = False
        orig_board = np.copy(board)

        if direction == 0:    # LEFT
            score += merge(board)
            pass
        elif direction == 1:  # RIGHT
            board = np.flip(board, axis=1)
            score += merge(board)
            board = np.flip(board, axis=1)
            pass
        elif direction == 2:  # UP
            board = np.transpose(board)
            score += merge(board)
            board = np.transpose(board)
            pass
        elif direction == 3:  # DOWN
            board = np.transpose(board)
            board = np.flip(board, axis=1)
            score += merge(board)
            board = np.flip(board, axis=1)
            board = np.transpose(board)
            pass
        else:
            print("ERR: Invalid Direction.")
            
        if np.array_equal(orig_board, board):
            invalid = True
            
        return score, invalid
    
def get_valid_actions(in_board):
    board = np.copy(in_board)
    valid_actions = []
    
    for action in range(4):
        _, invalid = move(board, action)
        if not invalid:
            valid_actions.append(action)
            
    return valid_actions

def one_hot_encode(board, board_size):
    log_values = np.log2(board[board != 0]).astype(np.float32)
    
    one_hot_encoded = np.zeros((1, board_size, board_size, (board_size**2)+1), dtype=np.int32)
    
    for i in range(len(log_values)):
        row, col = np.where(board == 2**log_values[i])
        one_hot_encoded[0, row, col, int(log_values[i])] = 1
        
    return one_hot_encoded

def preprocess_state(board):
    # log_values = np.log2(board[board != 0]).astype(np.float32)
    
    # normalized_board = np.zeros((4, 4))
    
    # for i in range(len(log_values)):
    #     row, col = np.where(board == 2**log_values[i])
    #     normalized_board[row, col] = log_values[i] / 16
    
    # return normalized_board
    return board

class Env:
    def __init__(self, board_size):
        self.board_size = board_size
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        place_rand_tile(self.board)
        place_rand_tile(self.board)
        self.game_over = False
        self.move_count = 0
        self.score = 0
        self.prev_score = 0
        self.max = 0
        self.prev_max = 0
        self.invalid_count = 0
        self.prev_invalid_count = 0
        self.zero_count = 0
        self.prev_zero_count = 0
        
        return self.board
        
    def step(self, action, count=True):
        self.prev_invalid_count = self.invalid_count
        self.prev_score = self.score
        self.prev_max = np.max(self.board)
        self.prev_zero_count = np.count_nonzero(self.board == 0)
        score_inc, self.invalid_count = move(self.board, action)
        self.score += score_inc
        reward = 0
        
        if self.invalid_count == self.prev_invalid_count:
            place_rand_tile(self.board)
            self.max = np.max(self.board)
            self.zero_count = np.count_nonzero(self.board == 0)
            self.update()
            reward = self.calculate_reward()
            if count:
                self.move_count += 1
                
        return self.board, reward, self.game_over
    
    def calculate_reward(self):
        
        # if (self.max == self.prev_max):
        #     return (self.zero_count - self.prev_zero_count)
        # else:
        #     return math.log((self.max-self.prev_max), 2) + (self.zero_count - self.prev_zero_count)
        if self.invalid_count == self.prev_invalid_count:
            return self.score - self.prev_score
        else:
            return -1000
                    
    def update(self):
        if np.any(self.board == 0):
            self.game_over = False
            return
        
        if np.any(np.diff(self.board, axis=1) == 0) or np.any(np.diff(self.board, axis=0) == 0):
            self.game_over = False
            return
            
        self.game_over = True
        
    def render(self):
        for row in self.board:
            format_row = ' '.join(f'{value:5}' for value in row)
            print(format_row, flush=True)
            
    def calculate_merges_value(self):
        merge_values = 0
        
        # for horizontal merges
        for row in range(self.board_size):
            squeezed_row = np.trim_zeros(self.board[row, :], 'b')
            for col in range(len(squeezed_row) - 1):
                if squeezed_row[col] == squeezed_row[col + 1] and squeezed_row[col] != 0:
                    merge_values += squeezed_row[col]
                    col += 1
                    
        # for vertical merges
        for col in range(self.board_size):
            squeezed_col = np.trim_zeros(self.board[:, col], 'b')
            for row in range(len(squeezed_col) - 1):
                if squeezed_col[row] == squeezed_col[row + 1] and squeezed_col[row] != 0:
                    merge_values += squeezed_col[row]
                    row += 1
                    
        return merge_values