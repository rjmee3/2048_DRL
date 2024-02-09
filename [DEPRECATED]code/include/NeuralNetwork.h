#ifndef NEURALNET_H
#define NEURALNET_H

typedef enum {
    SIGMOID,
    RELU,
    LINEAR,
    BINARY_STEP
} ActivationType;

typedef struct {
    double start_learning_rate = 0.0005;
    double gamma = 0.9;
    double epsilon = 0.9;
    int memory_capacity = 6000;
} hyperparams;

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

hyperparams->input_size;

/*  handles the allocation of memory for the network. Returns a neural network
    on success, otherwise returns NULL to indicate failure.     */
NeuralNetwork* create_network(int num_layers, int* layer_sizes, ActivationType* activations);
/*  handles the freeing of memory for the network. */
void destroy_network(NeuralNetwork* network);
/* conducts a forward pass through the network passed to it.    */
double* forward_pass(NeuralNetwork* network, double* input);
/*  function to train the neural network.   */
int train_network(NeuralNetwork* network, double* input, double learning_rate);

#endif