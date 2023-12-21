#include "GameState.h"
#include "Queue.h"
#include <stdio.h>
#include <string.h>

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
            row[index] += dequeue(&queue); 
        }

        index++;
    }

    // set all further elements equal to zero
    while (index < 4)
    {
        row[index] = 0;
        index++;
    }
}

void initializeGameState(GameState *state) {

    int testboard[4][4] = {
        {2,2,2,2},
        {0,0,0,0},
        {0,0,0,0},
        {0,0,0,0}
    };
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state->board[i][j] = testboard[i][j];
        }
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

            break;
        case MOVE_RIGHT:


            break;
        case MOVE_UP:


            break;
        case MOVE_DOWN:


            break;
        default:
            fprintf(stderr, "Invalid move.\n");
    }
}

int is_game_over(GameState *state) {
    
}

void boardToString(GameState *state, char *string) {
    string[0] = '\0';

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            sprintf(string + strlen(string), "%5d", state->board[i][j]);
        }
        strcat(string, "\n");
    }
}