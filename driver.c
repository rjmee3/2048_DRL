#include "GameState.h"
#include <stdio.h>

int main() {
    GameState state;

    initializeGameState(&state);

    char output[200];
    boardToString(&state, output);
    int move;

    printf("Initial Board:\n\n%s", output);

    while (1) {
        printf("\n\nEnter Move: ");
        scanf("%d", &move);

        switch (move) {
            case 1:
                apply_move(&state, MOVE_UP);
                break;

            case 2:
                apply_move(&state, MOVE_DOWN);
                break;

            case 3:
                apply_move(&state, MOVE_LEFT);
                break;

            case 4:
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