
# NOTE: You can only use Tensor API of PyTorch

import torch
from copy import deepcopy

# Extra TODO: Document with proper docstring
def mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr):
    """Mini-batch gradient descent
    """
    n_weights=deepcopy(weights)
    n_biases=deepcopy(biases)

    n_weights['w1'] -= (lr*dw1)
    n_weights['w2'] -= (lr*dw2)
    n_weights['w3'] -= (lr*dw3)
    n_biases['b1']  -= (lr*db1)
    n_biases['b2']  -= (lr*db2)
    n_biases['b3']  -= (lr*db3)


    return n_weights, n_biases

if __name__ == "__main__":
    pass
