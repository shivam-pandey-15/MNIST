
# NOTE: You can only use Tensor API of PyTorch

import random
import torch


model_dict = {}

def gen(x,y=1):
    a=[]
    for i in range(x*y):
        a.append(random.uniform(-1/(y**.5),1/(y**.5)))

    a=torch.tensor(a).view(x,y)
    return a.float()

def gen_b(x):
    a=[]
    for i in range(x):
        a.append((random.uniform(-1/(x**.5),1/(x**.5))))

    return torch.tensor(a).float()
class FullyConnected:
    """Constructs the Neural Network architecture.

    Args:
        N_in (int): input size
        N_h1 (int): hidden layer 1 size
        N_h2 (int): hidden layer 2 size
        N_out (int): output size
        device (str, optional): selects device to execute code. Defaults to 'cpu'

    Examples:
        >>> network = model.FullyConnected(2000, 512, 256, 5, device='cpu')
        >>> creloss, accuracy, outputs = network.train(inputs, labels)
    """

    def __init__(self, N_in, N_h1, N_h2, N_out, device='cpu'):
        """Initializes weights and biases, and construct neural network architecture.

        One [recommended](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) approach is to initialize weights randomly but uniformly in the interval from [-1/n^0.5, 1/n^0.5] where 'n' is number of neurons from incoming layer. For example, number of neurons in incoming layer is 784, then weights should be initialized randomly in uniform interval between [-1/784^0.5, 1/784^0.5].

        You should maintain a list of weights and biases which will be initalized here. They should be torch tensors.

        Optionally, you can maintain a list of activations and weighted sum of neurons in a dictionary named Cache to avoid recalculation of those. If tensors are too large it could be an issue.
        """
        self.N_in = N_in
        self.N_h1 = N_h1
        self.N_h2 = N_h2
        self.N_out = N_out

        self.device = torch.device(device)

        w1 =gen(N_h1,N_in)
        w2 =gen(N_h2,N_h1)
        w3 =gen(N_out,N_h2)
        self.weights = {'w1': w1, 'w2': w2, 'w3': w3}


        b1 = gen_b(N_h1)
        b2 = gen_b(N_h2)
        b3 = gen_b(N_out)
        self.biases = {'b1': b1, 'b2': b2, 'b3': b3}

        z1=  torch.zeros(256).float()
        z2=  torch.zeros(256).float()
        z3=  torch.zeros(10).float()

        self.cache = {'z1': z1, 'z2': z2, 'z3': z3}


    # TODO: Change datatypes to proper PyTorch datatypes

    def train(self, inputs, labels,lr=0.01, debug=True):
        """Trains the neural network on given inputs and labels.

        This function will train the neural network on given inputs and minimize the loss by backpropagating and adjusting weights with some optimizer.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in)
            labels (torch.tensor): correct labels. Size (batch_size)
            lr (float, optional): learning rate for training. Defaults to 0.001
            debug (bool, optional): prints loss and accuracy on each update. Defaults to False

        Returns:
            creloss (float): average cross entropy loss
            accuracy (float): ratio of correctly classified to total samples

        """


        outputs = self.forward(inputs)# forward pass

        creloss = loss.cross_entropy_loss(outputs,labels)  # calculate loss
        accuracy = self.accuracy(outputs,labels)# calculate accuracy

        if debug:
            print('loss: ', creloss)
            print('accuracy: ', accuracy)


        dw1, db1, dw2, db2, dw3, db3 = self.backward(inputs,labels,outputs)
        self.weights, self.biases = optimizer.mbgd(self.weights, self.biases, dw1, db1, dw2, db2, dw3, db3, lr)

        return creloss, accuracy , outputs

    def predict(self, inputs):
        """Predicts output probability and index of most activating neuron

        This function is used to predict output given inputs. You can then use index in classes to show which class got activated. For example, if in case of MNIST fifth neuron has highest firing probability, then class[5] is the label of input.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in)

        Returns:
            score (torch.tensor): max score for each class. Size (batch_size)
            idx (torch.tensor): index of most activating neuron. Size (batch_size)
        """
        outputs = self.forward(inputs)# forward pass

        score, idx = torch.max(outputs,1)# find max score and its index
        return score, idx

    def eval(self, inputs, labels,debug=True):
        """Evaluate performance of neural network on inputs with labels.

        This function is used to evaluate loss and accuracy of neural network on new examples. Unlike predict(), this function will not only predict but also calculate and return loss and accuracy w.r.t given inputs and labels.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in)
            labels (torch.tensor): correct labels. Size (batch_size)
            debug (bool, optional): print loss and accuracy on every iteration. Defaults to False

        Returns:
            loss (float): average cross entropy loss
            accuracy (float): ratio of correctly to uncorrectly classified samples
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """

        print("Train Data")
        outputs = self.forward(inputs)# forward pass



        creloss = loss.cross_entropy_loss(outputs,labels) # calculate loss
        accuracy = self.accuracy(outputs,labels)# calculate accuracy

        if debug:
            print('loss: ', creloss)
            print('accuracy: ', accuracy)

        return creloss, accuracy, outputs

    def accuracy(self, outputs, labels):
        """Accuracy of neural network for given outputs and labels.

        Calculates ratio of number of correct outputs to total number of examples.

        Args:
            outputs (torch.tensor): outputs predicted by neural network. Size (batch_size, N_out)
            labels (torch.tensor): correct labels. Size (batch_size)

        Returns:
            accuracy (float): accuracy score
        """

        score, idx = torch.max(outputs,1)

        count=0
        for i,j in zip(labels,idx):
            if i==j:
                count+=1

        accuracy = count/len(labels)
        return accuracy

    def forward(self, inputs):
        """Forward pass of neural network

        Calculates score for each class.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in)

        Returns:
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        self.cache['z1'] = self.weighted_sum(inputs.float(),self.weights['w1'],self.biases['b1']).float()
        a1 = activation.sigmoid(self.cache['z1']).float()
        self.cache['z2'] = self.weighted_sum(a1,self.weights['w2'],self.biases['b2'])
        a2 = activation.sigmoid(self.cache['z2']).float()
        self.cache['z3'] = self.weighted_sum(a2,self.weights['w3'],self.biases['b3'])
        outputs =  activation.softmax(self.cache['z3']).float()



        return outputs

    def weighted_sum(self, X, w, b):
        """Weighted sum at neuron

        Args:
            X (torch.tensor): matrix of Size (K, L)
            w (torch.tensor): weight matrix of Size (J, L)
            b (torch.tensor): vector of Size (J)

        Returns:
            r1esult (torch.tensor): w*X + b of Size (K, J)
        """

        result = torch.mm(X,w.t()) + b
        return result

    def backward(self, inputs, labels, outputs):
        """Backward pass of neural network

        Changes weights and biases of each layer to reduce loss

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in)
            labels (torch.tensor): correct labels. Size (batch_size)
            outputs (torch.tensor): outputs predicted by neural network. Size (batch_size, N_out)

        Returns:
            dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
            db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
            dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
            db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
            dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
            db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        """
        # Calculating derivative of loss w.r.t weighted sum




        dout = loss.delta_cross_entropy_softmax(outputs, labels)
        d2 =   (torch.mm(dout,self.weights['w3'])).float()*activation.delta_sigmoid(self.cache['z2'])
        d1 =   (torch.mm(d2,self.weights['w2'])).float()*activation.delta_sigmoid(self.cache['z1'])




        dw1, db1, dw2, db2, dw3, db3 = self.calculate_grad(inputs,d1,d2,dout)# calculate all gradients


        return dw1, db1, dw2, db2, dw3, db3

    def calculate_grad(self, inputs, d1, d2, dout):
        """Calculates gradients for backpropagation

        This function is used to calculate gradients like loss w.r.t. weights and biases.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in)
            dout (torch.tensor): error at output. Size like aout or a3 (or z3)
            d2 (torch.tensor): error at hidden layer 2. Size like a2 (or z2)
            d1 (torch.tensor): error at hidden layer 1. Size like a1 (or z1)

        Returns:
            dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
            db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
            dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
            db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
            dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
            db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        """
#        print("In Calc Grad\n-------------------------------------------------------")
        dw3 = torch.mm(dout.t(),activation.sigmoid(self.cache['z2']))
#        print(torch.sum(dw3),torch.sum(dout),torch.sum(activation.sigmoid(self.cache['z2'])),)
        dw2 = torch.mm(d2.t(),activation.sigmoid(self.cache['z1']))
#        print(torch.sum(dw2),torch.sum(d2),torch.sum(activation.sigmoid(self.cache['z1'])))
        dw1 = torch.mm(d1.t(),inputs.float())
#        print(torch.sum(dw1),torch.sum(d1),torch.sum(inputs))
#        print("In Calc Grad\n-------------------------------------------------------")
        db1 = torch.sum(d1,0)
        db2 = torch.sum(d2,0)
        db3 = torch.sum(dout,0)



        return dw1, db1, dw2, db2, dw3, db3


if __name__ == "__main__":
    import activation, loss, optimizer
else:
    from nnet import activation, loss, optimizer
