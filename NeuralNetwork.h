#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

typedef enum {
    BINARY_STEP,
    LINEAR,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    PARAMETRIC_RELU,
    ELU,
    SOFTMAX,
    SWISH,
    GELU,
    SELU
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

NeuralNetwork* create_network(int num_layers, int* layer_sizes, ActivationType* activations);

void destroy_network(NeuralNetwork* network);

double* forward_pass(NeuralNetwork* network, double* input);

void train_network_supervised(NeuralNetwork* network, double* input, double* target, double learning_rate);

void train_network_unsupervised(NeuralNetwork* network, double* input, double learning_rate);

#endif