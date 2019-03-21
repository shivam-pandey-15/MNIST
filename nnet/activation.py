
# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def sigmoid(z):
    """Calculates sigmoid values for tensors

    """
    result =  1/(1+torch.exp(-z))
    return result.float()

# Extra TODO: Document with proper docstring
def delta_sigmoid(z):
    """Calculates derivative of sigmoid function

    """
    grad_sigmoid = sigmoid(z).float()*(1-sigmoid(z)).float()
    '''
    print("---------------Grad Sigmoid--------------")
    print(z)
    print(grad_sigmoid)

    print("---------------Grad Sigmoid--------------")
    '''
    return grad_sigmoid

# Extra TODO: Document with proper docstring
def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors


    b=torch.max(x,1)
    b,c=b
    for i in range(len(x)):
        x[i]-=b[i]
    s=torch.sum(torch.exp(x),1)
    for i in range(len(x)):
        x[i]= torch.exp(x[i])/s[i]
    stable_softmax=x
    """
    x=x.exp()
    for i in range(len(x)):
        x[i]=x[i]/torch.sum(x[i])
    return x

if __name__ == "__main__":
    pass
