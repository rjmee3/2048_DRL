#include "GameState.h"
#include <stdio.h>

int main() {
    GameState state;

    initializeGameState(&state);

    char output[200];
    boardToString(&state, output);

    printf("Initial Board:\n\n%s", output);

    apply_move(&state, MOVE_LEFT);

    boardToString(&state, output);

    printf("New Board:\n\n%s", output);

    apply_move(&state, MOVE_LEFT);

    boardToString(&state, output);

    printf("New Board:\n\n%s", output);

    return 0;
}