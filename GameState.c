#include "GameState.h"
#include "Queue.h"
#include <stdio.h>

/*  Merges adjacent tiles with the same value to the left.  */
void merge_tiles_left(int row[4]) {
    // initialize a queue to store all values in one row
    Queue queue;
    initializeQueue(&queue);

    // enqueue all non-zero elements in the row
    for (int i = 0; i < 4; i++) {
        if (row[i] != 0) {
            enqueue(&queue, row[i]);
        }
    }

    int index = 0;

    // dequeue elements into the row, merging like elements
    while (!isEmpty(&queue)) {
        row[index] = dequeue(&queue);

        if (row[index] == front(&queue)) {
            row[index] =+ dequeue(&queue); 
        }

        index++;
    }
}

void apply_move(GameState *state, Action action) {
    // save an original copy of the state for comparison
    GameState original_state = *state;

    switch (action) {
        case MOVE_LEFT:
            for (int i = 0; i < 4; i++) {
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