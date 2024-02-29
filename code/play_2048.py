from state import State

if __name__ == "__main__":
    game = State(4)
    while not game.game_over:
        game.print_state()
        direction = input("Enter Move: ")
        
        if direction == "w":
            game.move_state(2)
        elif direction == "a":
            game.move_state(0)
        elif direction == "s":
            game.move_state(3)
        elif direction == "d":
            game.move_state(1)
        else:
            print("Invalid Input!")
        
        game.update_state()
        
    game.print_state()
    print("Game Over!")