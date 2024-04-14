# adding batches of inputs in the previous example

import numpy as np

# in this case let's take 3 batches of inputs.
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

'''
Notes:
As soon as the input is not a vector anymore, we will have dimension errors while
conducting dot products.

Therefore, to make the dimensions compartible, we do a dot product of input and transpose of weights:

OUTPUT = INPUT . T(WEIGHTS) + BIASES 

'''

output = np.dot(inputs, np.array(weights).T) + biases
# we put weights first (in the dot product) because we need outputs for three neurons.

print(output)


'''
outputs of layer 1 becomes inputs of layer 2
'''


'''
saving the model means we are actually saving the weights and biases.
two ways to initialize a layer:
1. load a model
2. new neural model (initialize weights and biases)
    weights between -0.1 to 0.1
    biases equal to 0
'''