from state import State

if __name__ == "__main__":
    state = State(4)
    while not state.game_over:
        state.print()
        direction = input("Enter Move: ")
        
        if direction == "w":
            state.move(2)
        elif direction == "a":
            state.move(0)
        elif direction == "s":
            state.move(3)
        elif direction == "d":
            state.move(1)
        else:
            print("Invalid Input!")
        
        state.update()
        
    state.print()
    print("Game Over!")