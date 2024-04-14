# adding another layer with 3 neurons in it as well

import numpy as np

# in this case let's take 3 batches of inputs.
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# first layer weights and biases
weights1 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2.0, 3.0, 0.5]

# second layer weights and biases
weights2 = [[0.1, -0.14, -0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1.0, 2.0, -0.5]

'''
Notes:
The layer 1 output becomes the inputs for the layer 2
'''


layer1_output = np.dot(inputs, np.array(weights1).T) + biases1
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print(layer2_output)


'''
outputs of layer 1 becomes inputs of layer 2
'''