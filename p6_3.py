# adding Softmax activation function class

'''
Notes:
Exponenentiation can result in huge values and sometimes an overflow beyond certain range of inputs.
Therefore,
From input, we subtract the max value from all the values before doing exponentiation.
This makes the largest value in the vector to a 0 and hence, output from the exponentiation always lies
between 0 to 1.

The final output after the normalization does not differ (with before and after this subtraction trick).
'''

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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

# Softmax activation class
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # subtraction trick to prevent overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities



X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

