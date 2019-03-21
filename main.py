
# Homecoming (eYRC-2018): Task 1A
# Build a Fully Connected 2-Layer Neural Network to Classify Digits

# NOTE: You can only use Tensor API of PyTorch

from nnet import model

# TODO: import torch and torchvision libraries
# We will use torchvision's transforms and datasets

import torch
import torchvision
from random import randint
from matplotlib import pyplot as plt



# TODO: Defining torchvision transforms for preprocessing
# TODO: Using torchvision datasets to load MNIST

train_data=torchvision.datasets.MNIST('./data', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]))


"""
Both the test and train dataset is being loaded and divided into batches
Batch Size for train is 4
Batch Size for test is 10000
Batch Size for predict is 10
"""
train_loader = torch.utils.data.DataLoader(train_data,batch_size=4, shuffle=True)


test_data=torchvision.datasets.MNIST('./data', train=False, download=True,
                                           transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]))



test_loader = torch.utils.data.DataLoader(test_data,batch_size=10000, shuffle=True)


# TODO: Use torch.utils.data.DataLoader to create loaders for train and test
# NOTE: Use training batch size = 4 in train data loader.


# NOTE: Don't change these settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# NOTE: Don't change these settings
# Layer size
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.1


# init model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

# TODO: Define number of epochs
N_epoch = 5 # Or keep it as is


# TODO: Training and Validation Loop
# >>> for n epochs
## >>> for all mini batches
### >>> net.train(...)
## at the end of each training epoch
## >>> net.eval(...)


#---------------------------------------------------------------------------------#
#----------------------------------TRAIN------------------------------------------#
#---------------------------------------------------------------------------------#


batch_size=4

train_loader=list(train_loader)

for i in range(len(train_loader)):
    train_loader[i][0]=train_loader[i][0].view(batch_size,-1)

accuracy = 0;
for epoch in range(N_epoch):
    print("Epoch ",epoch+1)
    l=[]
    a=[]
    for i in range(len(train_loader)):
        cressloss,acc,_=net.train(train_loader[i][0],train_loader[i][1],lr,False)
        l.append(cressloss)
        a.append(acc)
    total_loss=sum(l)/len(l)
    total_acc=sum(a)/len(a)
    print('loss: ', total_loss)
    print('accuracy: ', total_acc)
    if total_acc > accuracy:
        torch.save(net.state_dict(), 'MNIST.pth')


#---------------------------------------------------------------------------------#
#--------------------------------EVALUATE-----------------------------------------#
#---------------------------------------------------------------------------------#

batch_size_test=10000

test_loader=list(test_loader)


for i in range(len(test_loader)):
    test_loader[i][0]=test_loader[i][0].view(batch_size_test,-1)


for i in range(len(test_loader)):
    net.eval(test_loader[i][0],test_loader[i][1])

# TODO: End of Training
# make predictions on randomly selected test examples

#---------------------------------------------------------------------------------#
#--------------------------------PREDICT------------------------------------------#
#---------------------------------------------------------------------------------#

predict_loader = torch.utils.data.DataLoader(test_data,batch_size=10, shuffle=True)

batch_size_predict=10

predict_loader=list(predict_loader)


for i in range(len(predict_loader)):
    predict_loader[i][0]=predict_loader[i][0].view(batch_size_predict,-1)

a=randint(0,len(predict_loader))
prediction_v,pred=net.predict(predict_loader[a][0])
print(pred)






# >>> net.predict(...)
