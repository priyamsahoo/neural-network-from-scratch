# softmax activation function

'''
Notes:
Softmax activation function is used for the OUTPUT LAYER on CLASSIFCATION STYLE NUERAL NETWORKS.

- training model - determining how wrong the model prediction is
- ReLU activatoion function - it is exclusive, per neuron basis, no relative comparison between neurons

- this is why need softmax activation function

softmax activation function: exponentiating and normalizing

input -> exponentiate -> normalize -> output

'''

import numpy as np
import nnfs

nnfs.init()

layer_output = [4.8, 1.21, 2.385]

# E = 2.71828182846

exp_values = np.exp(layer_output)
norm_values = exp_values / np.sum(exp_values)

print(norm_values)