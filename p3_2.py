# coding (modeling) a layer consisting of three neurons from a neural network
# make the code more dynamic (part 2 - using numpy)

import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

'''
Notes:
concept of shape of a list, eg:
    1. [1, 2, 3, 4] -> shape = (4,)

    2. [[1, 2, 3, 4],
        [2, 3, 4, 4],
        [5, 6, 7, 8]] -> shape = (3, 4)

    3. [[[1, 2, 3, 4],
        [2, 3, 4, 4],
        [5, 6, 7, 8]],

        [[1, 2, 2, 3],
        [1, 2, 2, 3],
        [5, 6, 7, 2]]] -> shape = (2, 3, 4)
'''

output = np.dot(weights, inputs) + biases
# we put weights first (in the dot product) because we need outputs for three neurons.

print(output)