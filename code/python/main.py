from game_state import print_board, initialize_board, move, is_game_over

if __name__ == "__main__":
    
    board = initialize_board()
    while not is_game_over(board):
        print_board(board)
        direction = input("Enter Move: ")
        board = move(board, direction)
        
    print_board(board)
    print("Game Over!")