from collections import deque
import copy
import random

# program constants
BOARD_SIZE = 4

# function to print the board to the console
def print_board(board):
    for row in board:
        format_row = ' '.join(f'{value:5}' for value in row)
        print(format_row, flush=True)
        
# function to initialize the board 
def initialize_board():
    board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    place_rand_tile(board)
    place_rand_tile(board)
    return board

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
        return board
    if board != orig_board:
        place_rand_tile(board)

    return board

# function to transpose board matrix.
def transpose(board):
    return [list(row) for row in zip(*board)]

# function to reverse board matrix.
def reverse(board):
    return [row[::-1] for row in board]

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
    for row in board:
        if 0 in row or any(row[i] == row[i+1] for i in range(BOARD_SIZE-1)):
            return False
        
    for col in range(len(board[0])):
        col_val = [board[row][col] for row in range(BOARD_SIZE)]
        if 0 in col_val or any(col_val[i] == col_val[i+1] for i in range(BOARD_SIZE-1)):
            return False
        
    return True