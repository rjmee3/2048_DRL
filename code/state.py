from collections import deque
import numpy as np
import random
import torch

BOARD_SIZE = 4

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
def merge(board, score, prev_score):
    queue = deque()
    prev_score = prev_score
    
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
                prev_score = score
                score += row[index]
            
            index += 1
            
        # set all further elements to zero
        while index < len(row):
            row[index] = 0
            index += 1

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
    
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
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
    
    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
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
            
    def move(self, direction):
        orig_board = np.copy(self.board)
        if direction == 0:    # LEFT
            merge(self.board, self.score, self.prev_score)
            self.left_count += 1
            pass
        elif direction == 1:  # RIGHT
            self.board = np.flip(self.board, axis=1)
            merge(self.board, self.score, self.prev_score)
            self.board = np.flip(self.board, axis=1)
            self.right_count += 1
            pass
        elif direction == 2:  # UP
            self.board = np.transpose(self.board)
            merge(self.board, self.score, self.prev_score)
            self.board = np.transpose(self.board)
            self.up_count += 1
            pass
        elif direction == 3:  # DOWN
            self.board = np.transpose(self.board)
            self.board = np.flip(self.board, axis=1)
            merge(self.board, self.score, self.prev_score)
            self.board = np.flip(self.board, axis=1)
            self.board = np.transpose(self.board)
            self.down_count += 1
            pass
        else:
            print("ERR: Invalid Action.")
            
        if not np.array_equal(self.board, orig_board):
            place_rand_tile(self.board)
        else:
            self.invalid_count += 1
            
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
            print(format_row)  
        
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
    
    def one_hot_encode(self, batch_size=1):
        log_values = torch.log2(self.board[self.board != 0], dtype=torch.float32)
        
        one_hot_encoded = torch.zeros((batch_size, BOARD_SIZE, BOARD_SIZE, (BOARD_SIZE**2)+1), dtype=torch.int)
        
        for i in range(len(log_values)):
            row, col = torch.where(torch.tensor(self.board) == 2**log_values[i])
            one_hot_encoded[:, row, col, int(log_values[i])] = 1
            
        return one_hot_encoded
    
    def calculate_reward(self):
        return self.score - self.prev_score