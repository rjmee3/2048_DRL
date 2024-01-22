#ifndef Q_LEARNING_H
#define Q_LEARNING_H

#include "GameState.h"

typedef struct {
    int num_states;
    int num_actions;
    double learning_rate;
    double discount_factor;
    double exploration_rate;
    double*** q_table;
} QLearning;

QLearning* create_q_learning(int num_states, int num_actions, double learning_rate, double discount_factor, double exploration_rate);

void destroy_q_learning(QLearning* ql);

void update_q_learning(QLearning* ql, GameState* state, Action action, double reward, GameState* next_state);

Action choose_action(QLearning* ql, GameState* state);

#endif