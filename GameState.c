#include "GameState.h"
#include <stdio.h>

// /*  moves non-zero tiles to the left.   */
// void shift_tiles_left(int row[4]) {
//     // enqueue each element within the row
//     for (int i = 0; i < 4; i++) {
//         if (row[i] != 0) {
            
//         }
//     }
// }

/*  Merges adjacent tiles with the same value to the left.  */
void merge_tiles_left(int row[4]) {

}

void apply_move(GameState *state, Action action) {
    // save an original copy of the state for comparison
    GameState original_state = *state;

    switch (action) {
        case MOVE_LEFT:
            for (int i = 0; i < 4; i++) {
                // shift_tiles_left(state->board[i]);
                merge_tiles_left(state->board[i]);
            }

        case MOVE_RIGHT:


        case MOVE_UP:


        case MOVE_DOWN:


        default:
            fprintf(stderr, "Invalid move.\n");
    }
}

int is_game_over(GameState *state) {
    
}