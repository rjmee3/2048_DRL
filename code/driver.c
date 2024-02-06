#include "include/GameState.h"
#include <stdio.h>

int main() {
    GameState state;

    initializeGameState(&state);

    char output[200];
    boardToString(&state, output);
    char move;

    printf("Initial Board:\n\n%s", output);

    while (1) {
        if (is_game_over(&state)) {
            printf("GAME OVER\n");

            break;
        }

        printf("\n\nEnter Move: ");
        scanf(" %c", &move);

        switch (move) {
            case 'w':
                apply_move(&state, MOVE_UP);
                break;

            case 's':
                apply_move(&state, MOVE_DOWN);
                break;

            case 'a':
                apply_move(&state, MOVE_LEFT);
                break;

            case 'd':
                apply_move(&state, MOVE_RIGHT);
                break;

            default:
                printf("\n\nInvalid Move...\n\n");
        }

        boardToString(&state, output);

        printf("New Board:\n\n%s", output);
    }

    return 0;
}