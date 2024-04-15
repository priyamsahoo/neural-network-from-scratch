# softmax activation function on a batch of output

'''
Notes:
INPUTS here are the outputs so far of the model.

'''

import numpy as np
import nnfs

nnfs.init()

layer_output = [[4.8, 1.21, 2.385],
                [8.9, -1.82, 0.2],
                [1.41, 1.051, 0.026]]


exp_values = np.exp(layer_output)

# since 2D matrix, axis = 0 means column-wise sum, axis = 1 means row-wise sum.
# keepdims = True maintains the 2D structure but has only one value in each row that is the sum of that row
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)