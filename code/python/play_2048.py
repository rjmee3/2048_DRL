from game import Game

if __name__ == "__main__":
    game = Game(4)
    while not game.game_over:
        game.print_board()
        direction = input("Enter Move: ")
        game.move(direction)
        game.update()
        
    game.print_board()
    print("Game Over!")