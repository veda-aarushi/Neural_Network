//
// Created by vedaa on 3/20/2025.
//

#include "Neural_Network.h"
// neural_network.cpp
// Fully connected neural network from scratch in C++

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Activation functions
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

float tanh_activation(float x) {
    return tanh(x);
}

float tanh_derivative(float x) {
    return 1 - x * x;
}

vector<float> softmax(vector<float> &inputs) {
    float sum_exp = 0.0f;
    for (float val : inputs) sum_exp += exp(val);
    vector<float> output;
    for (float val : inputs) output.push_back(exp(val) / sum_exp);
    return output;
}

// Neural Network class
class NeuralNetwork {
private:
    vector<vector<float>> weights_input_hidden;
    vector<float> biases_hidden;
    vector<float> weights_hidden_output;
    float bias_output;
    int input_nodes, hidden_nodes, output_nodes;

public:
    NeuralNetwork(int input, int hidden, int output) {
        input_nodes = input;
        hidden_nodes = hidden;
        output_nodes = output;
        srand(time(0));

        weights_input_hidden.resize(input, vector<float>(hidden));
        biases_hidden.resize(hidden);
        weights_hidden_output.resize(hidden);
        bias_output = ((float)rand() / RAND_MAX) * 2 - 1;

        for (int i = 0; i < input; i++) {
            for (int j = 0; j < hidden; j++) {
                weights_input_hidden[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
            }
        }

        for (int j = 0; j < hidden; j++) {
            biases_hidden[j] = ((float)rand() / RAND_MAX) * 2 - 1;
            weights_hidden_output[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }

    float feedforward(vector<float> inputs) {
        vector<float> hidden_layer(hidden_nodes);
        for (int j = 0; j < hidden_nodes; j++) {
            hidden_layer[j] = biases_hidden[j];
            for (int i = 0; i < input_nodes; i++) {
                hidden_layer[j] += inputs[i] * weights_input_hidden[i][j];
            }
            hidden_layer[j] = tanh_activation(hidden_layer[j]);
        }

        float output = bias_output;
        for (int j = 0; j < hidden_nodes; j++) {
            output += hidden_layer[j] * weights_hidden_output[j];
        }
        return sigmoid(output);
    }

    void train(vector<float> inputs, float target, float learning_rate) {
        // Forward pass
        vector<float> hidden_layer(hidden_nodes);
        for (int j = 0; j < hidden_nodes; j++) {
            hidden_layer[j] = biases_hidden[j];
            for (int i = 0; i < input_nodes; i++) {
                hidden_layer[j] += inputs[i] * weights_input_hidden[i][j];
            }
            hidden_layer[j] = tanh_activation(hidden_layer[j]);
        }

        float output = bias_output;
        for (int j = 0; j < hidden_nodes; j++) {
            output += hidden_layer[j] * weights_hidden_output[j];
        }
        output = sigmoid(output);

        // Backpropagation
        float output_error = target - output;
        float output_gradient = output_error * sigmoid_derivative(output) * learning_rate;

        vector<float> hidden_errors(hidden_nodes);
        for (int j = 0; j < hidden_nodes; j++) {
            hidden_errors[j] = output_gradient * weights_hidden_output[j] * tanh_derivative(hidden_layer[j]);
        }

        // Update weights and biases
        for (int j = 0; j < hidden_nodes; j++) {
            weights_hidden_output[j] += hidden_layer[j] * output_gradient;
        }
        bias_output += output_gradient;

        for (int j = 0; j < hidden_nodes; j++) {
            for (int i = 0; i < input_nodes; i++) {
                weights_input_hidden[i][j] += inputs[i] * hidden_errors[j];
            }
            biases_hidden[j] += hidden_errors[j];
        }
    }
};

int main() {
    NeuralNetwork nn(2, 6, 1); // Increased hidden neurons

    // XOR training data
    vector<vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<float> targets = {0, 1, 1, 0};

    // Training loop with mini-batch approach
    for (int i = 0; i < 50000; i += 4) {
        float learning_rate = 0.1f / (1.0f + 0.001f * i); // Decaying learning rate
        for (int j = 0; j < 4; ++j) {
            nn.train(inputs[j], targets[j], learning_rate);
        }
    }

    // Testing
    for (int i = 0; i < 4; ++i) {
        float output = nn.feedforward(inputs[i]);
        cout << "Input: " << inputs[i][0] << ", " << inputs[i][1] << " -> Output: " << round(output) << endl;
    }
    return 0;
}
