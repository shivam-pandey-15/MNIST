
# NOTE: You can only use Tensor API of PyTorch

import torch,numpy as np

from nnet import activation
from math import e
# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels




    print("These are labels\n\n")
    print(labels,'\n\n',outputs)
    """
    l=torch.zeros(outputs.shape).float()
    l[range(len(labels)),labels]=1
    labels=l
    s=e**-15
    outputs.clamp(s,1-s)


    creloss= -torch.sum(labels*torch.log(outputs))/labels.shape[0]


    return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z).

    """

    m = labels.shape[0]

    

    outputs[range(m),labels] -= 1
    outputs/=len(labels)

    return outputs

if __name__ == "__main__":
    pass
