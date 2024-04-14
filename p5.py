# activation function
# produce activation for a an entire single layer of neurons

'''
Notes:
Every neuron in the hidden and the output layer will have an activation function associated.
Generally, the output layer will have a different activation function than the hidden layers

Activation function comes in after { INPUT x WEIGHT + BIAS }

1. step function - 0,1 output
2. sigmoid function - more granular output from the function
3. rectified linear unit function (ReLU) - granular and fast (faster than signmoid)

Why activation function?
- without it, the output is simply linear, i.e., y = mx + c
- thus, the neural network can only be used to fit a linear function.
- in attempts to fit this to a non-linear function (eg. sin wave), we can only
    approximate with a linear function.

- this is why we use an activation function (for the case of sin, we use a sigmoid or ReLU ac. func.)
to fit the neural network to a non-linear function.

    
- in the example below, we have the same activation function for the entire layer, but 
we don't actually have to have the same activation function in the entire layer. They could be mixed.
'''

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# dataset of 100 feature sets of 3 classes
X, y = spiral_data(100, 3)

# Layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# layer object with 2 unique features as inputs and 5 neurons
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
# print(layer1.output)

# pass the output of each neuron from layer 1 to the activation function
activation1.forward(layer1.output)
print(activation1.output)
