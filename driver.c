#include "GameState.h"
#include <stdio.h>

int main() {
    GameState state;

    initializeGameState(&state);

    char output[200];
    boardToString(&state, output);

    printf("Initial Board:\n\n%s", output);

    apply_move(&state, MOVE_UP);
    boardToString(&state, output);
    printf("Move Up.\nNew Board:\n\n%s", output);

    apply_move(&state, MOVE_RIGHT);
    boardToString(&state, output);
    printf("Move Right.\nNew Board:\n\n%s", output);

    apply_move(&state, MOVE_DOWN);
    boardToString(&state, output);
    printf("Move Down.\nNew Board:\n\n%s", output);

    apply_move(&state, MOVE_LEFT);
    boardToString(&state, output);
    printf("Move Left.\nNew Board:\n\n%s", output);

    return 0;
}