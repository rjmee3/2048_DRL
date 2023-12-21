#ifndef GAMESTATE_H
#define GAMESTATE_H

/*  Structure which defines a game state.
    4x4 board represents the values of each tile at a point in time.
    Score represents the value of the board as a single integer value.  */
typedef struct {
    int board[4][4];
    int score;
} GameState;

/*  The defined set of actions which can be taken.  */
typedef enum {
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    MOVE_DOWN
} Action;

/*  initializes game state with 2 random tiles  */
void initializeGameState(GameState *state);
/*  Applies the action to the game state.*/
void apply_move(GameState *state, Action action);
/*  Checks the game state passed for whether it is possible
    to make another move. Returns 0 if another move is available,
    returns 1 if there are no moves left, indicating a game over.   */
int is_game_over(GameState *state);
/*  Formats board into string.      */
void boardToString(GameState *state, char *string);

#endif