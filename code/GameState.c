#include "include/GameState.h"
#include "include/Queue.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/*  Merges adjacent tiles with the same value to the left.  */
void merge_tiles(int row[4]) {
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

        // this merges like elements
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

/*  function to transpose board matrix for up and down moves    */
void transpose(int original_matrix[4][4]) {
    int temp_matrix[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp_matrix[j][i] = original_matrix[i][j];
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            original_matrix[i][j] = temp_matrix[i][j];
        }
    }
}

/*  function to reflect board matrix    */
void reflect(int original_matrix[4][4]) {
    int temp_matrix[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp_matrix[i][3-j] = original_matrix[i][j];
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            original_matrix[i][j] = temp_matrix[i][j];
        }
    }
}

/*  Spawns a random tile on the board   */
void spawnTile(GameState *state) {

    srand(time(0));

    int empty_space = 0;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (state->board[i][j] == 0) {
                empty_space++;
            }
        }
    }

    if (empty_space == 0) {
        return;
    }

    int rand_space = empty_space > 1 ? rand() % (empty_space) : 0;

    int index = 0;

    int rnum = (rand() % 10) == 0 ? 4 : 2;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (state->board[i][j] == 0 && rand_space == index++) {
                state->board[i][j] = rnum;
                return;
            }
        }
    }
}

void initializeGameState(GameState *state) {

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state->board[i][j] = 0;
        }
    }
    
    spawnTile(state);
    spawnTile(state);
}

int apply_move(GameState *state, Action action) {
    // save an original copy of the state for comparison
    GameState original_state = *state;

    char output[200];

    switch (action) {
        case MOVE_LEFT:
            for (int i = 0; i < 4; i++) {
                merge_tiles(state->board[i]);
            }

            break;

        case MOVE_RIGHT:
            reflect(state->board);

            for (int i = 0; i < 4; i++) {
                merge_tiles(state->board[i]);
            }
            
            // undo transformations
            reflect(state->board);

            break;

        case MOVE_UP:
            transpose(state->board);

            for (int i = 0; i < 4; i++) {
                merge_tiles(state->board[i]);
            }

            // undo transformations
            transpose(state->board);

            break;

        case MOVE_DOWN:
            transpose(state->board);
            reflect(state->board);

            for (int i = 0; i < 4; i++) {
                merge_tiles(state->board[i]);
            }

            // undo transformations
            reflect(state->board);
            transpose(state->board);

            break;

        default:
            fprintf(stderr, "Invalid move.\n");
    }

    if (memcmp(state, &original_state, sizeof(GameState)) != 0) {
        spawnTile(state);

        return 0;
    }

    return 1;
}

int is_game_over(GameState *state) {
    // check for any tiles which contain 0
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (state->board[i][j] == 0) {
                return 0;
            }
        }
    }

    // check for available merges
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (state->board[i][j] == state->board[i+1][j] || state->board[i][j] == state->board[i][j+1]) {
                return 0;
            }
        }
    }

    for (int i = 0; i < 3; i++) {
        if (state->board[3][i] == state->board[3][i+1]) {
            return 0;
        }

        if (state->board[i][3] == state->board[i+1][3]) {
            return 0;
        }
    }

    return 1;
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