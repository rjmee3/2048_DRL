from collections import deque
import numpy as np
import random

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
def merge(board):
    queue = deque()
    
    for row in board:
        for i in range(len(row)):
            if row[i] != 0:
                queue.append(row[i])
                
        index = 0
        while queue:
            row[index] = queue.popleft()
            
            if queue and row[index] == queue[0]:
                row[index] += queue.popleft()
            
            index += 1
            
        while index < len(row):
            row[index] = 0
            index += 1

    return board

class State: 
    # function to initialize the board 
    def __init__(self, board_size):
        self.board = np.zeros((board_size, board_size), dtype=int)
        place_rand_tile(self.board)
        place_rand_tile(self.board)
        self.game_over = False
        self.move_count = 0
        self.invalid_count = 0
        self.up_count = 0
        self.down_count = 0
        self.left_count = 0
        self.right_count = 0
    
    # function to print the board to the console
    def print_state(self):
        for row in self.board:
            format_row = ' '.join(f'{value:5}' for value in row)
            print(format_row)  
            
    def move_state(self, direction):
        orig_board = np.copy(self.board)
        if direction == 0:    # LEFT
            merge(self.board)
            self.left_count += 1
            pass
        elif direction == 1:  # RIGHT
            self.board = np.flip(self.board, axis=1)
            merge(self.board)
            self.board = np.flip(self.board, axis=1)
            self.right_count += 1
            pass
        elif direction == 2:  # UP
            self.board = np.transpose(self.board)
            merge(self.board)
            self.board = np.transpose(self.board)
            self.up_count += 1
            pass
        elif direction == 3:  # DOWN
            self.board = np.transpose(self.board)
            self.board = np.flip(self.board, axis=1)
            merge(self.board)
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
    def update_state(self):
        if np.any(self.board == 0):
            self.game_over = False
            return
        
        if np.any(np.diff(self.board, axis=1) == 0) or np.any(np.diff(self.board, axis=0) == 0):
            self.game_over = False
            return
            
        self.game_over = True