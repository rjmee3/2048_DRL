#ifndef NEURALNET_H
#define NEURALNET_H

typedef enum {
    SIGMOID,
    RELU,
    LINEAR,
    BINARY_STEP
} ActivationType;

typedef struct {
    int input_size;
    int output_size;
    double** weights;
    double* biases;
    ActivationType activation;
} Layer;

typedef struct {
    int num_layers;
    Layer* layers;
} NeuralNetwork;

/*  handles the allocation of memory for the network. Returns a neural network
    on success, otherwise returns NULL to indicate failure.     */
NeuralNetwork* create_network(int num_layers, int* layer_sizes, ActivationType* activations);
/*  handles the freeing of memory for the network. */
void destroy_network(NeuralNetwork* network);

double* forward_pass(NeuralNetwork* network, double* input);

int train_network(NeuralNetwork* network, double* input, double learning_rate);

#endif