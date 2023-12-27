#include "NeuralNetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

NeuralNetwork* create_network(int num_layers, int* layer_sizes, ActivationType* activations) {
    // allocate memory for neural network
    NeuralNetwork* network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (network == NULL) {
        return NULL;
    }

    // set number of layers in network
    network->num_layers = num_layers;

    // allocate memory for layers
    network->layers = (Layer*)malloc(num_layers * sizeof(Layer));
    if (network->layers == NULL) {
        free(network);
        return NULL;
    }

    // initialize each layer
    for (int i = 0; i < num_layers; i++) {
        network->layers[i].input_size = i == 0 ? layer_sizes[i] : layer_sizes[i - 1];
        network->layers[i].output_size = layer_sizes[i];
        network->layers[i].activation = activations[i];

        // allocate memory for weight matrix
        network->layers[i].weights = (double**)malloc(network->layers[i].output_size * sizeof(double*));
        if (network->layers[i].weights == NULL) {
            destroy_network(network);
            return NULL;
        }

        for (int j = 0; j < network->layers[i].output_size; j++) {
            network->layers[i].weights[j] = (double*)malloc(network->layers[i].input_size * sizeof(double));
            if (network->layers[i].weights[j] == NULL) {
                destroy_network(network);
                return NULL;
            }

            // init weights (defaulting to zero, might change later)
            for (int k = 0; k < network->layers[i].input_size; k++) {
                network->layers[i].weights[j][k] = 0;
            }
        }

        // allocate memory for bias vector
        network->layers[i].biases = (double*)malloc(network->layers[i].output_size * sizeof(double));
        if (network->layers[i].biases == NULL) {
            destroy_network(network);
            return NULL;
        }

        // init biases (defaulting to zero, might change later)
        for (int j = 0; j < network->layers[i].output_size; j++) {
            network->layers[i].biases[j] = 0;
        }
    }

    return network;
}

void destroy_network(NeuralNetwork* network) {
    // early return if network is unallocated
    if (network == NULL) { 
        return;
    }

    // free memory from each layer
    for (int i = 0; i < network->num_layers; i++) {
        // free weight matrix
        for (int j = 0; j < network->layers[i].output_size; j++) {
            free(network->layers[i].weights[j]);
        }
        free(network->layers[i].biases);

        // free bias vector
        free(network->layers[i].weights);
        
    }

    // free layers array
    free(network->layers);

    // finally, free network
    free(network);
}

double apply_activation_function(ActivationType activation, double x) {
    /*  commented out functions require more than one variable, 
        and would require some major refactoring. therefore, I 
        have decided to just not address the problem and leave 
        it for further down the line (when it is inevitably
        going to be an even bigger problem to fix). :)
    */
    // switch based on activation function type
    switch (activation) {
    case BINARY_STEP:
        return (x >= 0) ? 1.0 : 0.0;

    case LINEAR:
        return x;
    
    case SIGMOID:
        return 1.0 / (1.0 + exp(-x));

    case TANH:
        return tanh(x);

    case RELU:
        return (x > 0) ? x : 0;

    /*
    case LEAKY_RELU:
        return;

    case PARAMETRIC_RELU:
        return;

    case ELU:
        return;

    case SOFTMAX:
        return;

    */

    case SWISH:
        return x * (1.0 / (1.0 + exp(-x)));

    case GELU:
        return 0.5 * x * (1 + tanh(M_SQRT1_2 * (x + 0.044715 * pow(x, 3))));

    /*
    case SELU:
        return;
    */
    }
}

double* forward_pass(NeuralNetwork* network, double* input) {
    // early return if network or input is unallocated
    if (network == NULL || input == NULL) {
        return NULL;
    }

    // allocate memory for output vector
    double* output = (double*)malloc(network->layers[network->num_layers - 1].output_size * sizeof(double));
    if (output == NULL) {
        return NULL;
    }

    // init input vector for first layer
    double* layer_input = input;
    

    // perform forward pass through each layer
    for (int i = 0; i < network->num_layers; i++) {
        // allocate memory for output vector of *current* layer
        double* layer_output = (double*)malloc(network->layers[i].output_size * sizeof(double));
        if (layer_output == NULL) {
            free(output);
            return NULL;
        }

        // calculate weighted sum and apply activation function
        for (int j = 0; j < network->layers[i].output_size; j++) {
            layer_output[j] = 0;
            for (int k = 0; k < network->layers[i].input_size; k++) {
                layer_output[j] += layer_input[k] * network->layers[i].weights[j][k];
            }
            layer_output[j] += network->layers[i].biases[j];

            layer_output[j] = apply_activation_function(network->layers[i].activation, layer_output[j]);
        }

        // free memory for input vector
        free(layer_input);

        layer_input = layer_output;
    }

    // copy output to allocated memory
    for (int i = 0; i < network->layers[network->num_layers - 1].output_size; i++) {
        output[i] = layer_input[i];
    }

    return output;
}

void train_network_supervised(NeuralNetwork* network, double* input, double* target, double learning_rate) {
    // early return if network or input is unallocated
    if (network == NULL || input == NULL || target == NULL) {
        return;
    }

    // perform forward pass to get predicted output
    double* output = forward_pass(network, input);
    if (output == NULL) {
        return;
    }

    // perform backward pass to update weights and biases
    for (int i = network->num_layers - 1; i >= 0; i--) {
        // calculate error for output layer
        double* error = (double*)malloc(network->layers[i].output_size * sizeof(double));
        if (error == NULL) {
            free(output);
            return;
        }
        if (i == network->num_layers - 1) {
            // error for output layer
            for (int j = 0; j < network->layers[i].output_size; j++) {
                error[j] = target[j] - output[j];
            }
        } else {
            // error for hidden layers
            for (int j = 0; j < network->layers[i].output_size; j++) {
                error[j] = 0;
                for (int k = 0; k < network->layers[i + 1].output_size; k++) {
                    error[j] += network->layers[i + 1].weights[k][j] * error[k];
                }
            }
        }

        // update weights and biases for reconstruction
        for (int j = 0; j < network->layers[i].output_size; j++) {
            // update biases
            network->layers[i].biases[j] += learning_rate * error[j];

            // update weights
            for (int k = 0; k < network->layers[i].input_size; k++) {
                network->layers[i].weights[j][k] += learning_rate * error[j] * input[k];
            }
        }

        // free error vector
        free(error);
    }

    // free output vector
    free(output);
}

void train_network_unsupervised(NeuralNetwork* network, double* input, double learning_rate) {
    // early return if network or input is unallocated
    if (network == NULL || input == NULL) {
        return;
    }

    // perform forward pass to get predicted output
    double* output = forward_pass(network, input);
    if (output == NULL) {
        return;
    }

    // perform backward pass to update weights and biases
    for (int i = network->num_layers - 1; i >= 0; i--) {
        // calculate error for output layer
        double* error = (double*)malloc(network->layers[i].output_size * sizeof(double));
        if (error == NULL) {
            free(output);
            return;
        }
        if (i == network->num_layers - 1) {
            // error for output layer
            for (int j = 0; j < network->layers[i].output_size; j++) {
                error[j] = input[j] - output[j];
            }
        } else {
            // error for hidden layers
            for (int j = 0; j < network->layers[i].output_size; j++) {
                error[j] = 0;
                for (int k = 0; k < network->layers[i + 1].output_size; k++) {
                    error[j] += network->layers[i + 1].weights[k][j] * error[k];
                }
            }
        }

        // update weights and biases for reconstruction
        for (int j = 0; j < network->layers[i].output_size; j++) {
            // update biases
            network->layers[i].biases[j] += learning_rate * error[j];

            // update weights
            for (int k = 0; k < network->layers[i].input_size; k++) {
                network->layers[i].weights[j][k] += learning_rate * error[j] * input[k];
            }
        }

        // free error vector
        free(error);
    }

    // free output vector
    free(output);
}