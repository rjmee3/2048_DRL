#include "GameState.h"

int main() {
    GameState state;

    initializeGameState(&state);

    char output[1000];
    boardToString(&state, output);

    printf("Initial Board:\n\n%s", output);

    return 0;
}