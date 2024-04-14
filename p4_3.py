# creating a class of to convert the concept of layers to objects

import numpy as np

np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]


'''
Notes:
    - saving the model means we are actually saving the weights and biases.
    - two ways to initialize a layer:
        1. load a model
        2. new neural model (initialize weights and biases)
    
    - initialization:
        1. weights between -0.1 to 0.1
        2. biases equal to 0
'''

# Layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases

# Layer objects
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)
