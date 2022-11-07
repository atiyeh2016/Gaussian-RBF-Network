#%% Importing Libraries
import numpy as np
import gzip, pickle #To load MNIST dataset
import time #To measure time
import matplotlib.pyplot as plt #To plot results
from keras.utils import to_categorical

#%% Defining Hyperparameters
num_data = 50000
lr = 1e-3
epochs = 300
batch = 128
im_size = 28*28
structure = [im_size,150,10]
num_in = structure[0]
num_out = structure[-1]
num_h_layer = len(structure) - 2 #number of hidden layers

#%% Seed
#Using seed for random to have consistant output
np.random.seed(1234567)

#%% Loading MNIST and Preparing Data
time_0 = time.time()

#Loading MNIST dataset from file
with gzip.open('mnist.pkl.gz','rb') as f :
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data_train, data_val, data_test = u.load()

time_1 = time.time()

#Seperating training data to data and labels
data = data_train[0]
labels = data_train[1]
data = data[0:num_data]
labels = labels[0:num_data]

#Plot the first image of MNIST dataset
#plt.imshow(data[0].reshape(28,28))

#Measuring the mean and make the data zero-centered
mean = np.mean(data, axis=0)
for i in range(data.shape[0]):
    data[i,:] -= mean

labels_cat = to_categorical(labels)
labels_cat_ = labels_cat.copy()
labels_cat_[labels_cat_==0] = -1

#%% Initialization
W, B, A, Z = [], [], [], []
dA, dZ, dW, dB = [], [], [], []

for i in range(num_h_layer + 1):
    W.append(np.random.rand(structure[i+1],structure[i])/100)
    dW.append(np.zeros([structure[i+1],structure[i]]))
    B.append(np.random.rand(structure[i+1],1)/100)
    dB.append(np.zeros([structure[i+1],1]))

for i in range(num_h_layer + 2):
    A.append(np.zeros([structure[i],1]))
    Z.append(np.zeros([structure[i],1]))
    dA.append(np.zeros([structure[i],1]))
    dZ.append(np.zeros([structure[i],1]))

#%% Loss Defenition
def hinge_loss(T, Y):
    Y = Y.reshape(10,1)
    T = T.reshape(10,1)
    return sum(np.multiply((np.multiply(T,Y)<1),1-np.multiply(Y,T)))

#%% Training
L = np.zeros([1,epochs])
for epoch in range(epochs):
    lim1 = 0
    lim2 = batch
    t_0 = time.time()
    while lim2 < num_data:
        data_batch = data[lim1:lim2]
        labels_batch_cat_=labels_cat_[lim1:lim2]
        labels_batch_cat = labels_cat[lim1:lim2]
        lim1 += batch
        lim2 += batch
        
        for k in range(len(data_batch)):
            
            # Forward:
            x = data_batch[k].reshape(im_size,1)#.astype('float64')
            Z[0] = x #.copy() ?
            A[0] = x
            for i in range(num_h_layer + 1):
                Z[i+1] = np.matmul(W[i],A[i]) + B[i]
                A[i+1] = np.multiply(Z[i+1],(Z[i+1]>0)) #relu function
            A[-1] = Z[-1]
            
            # Backward
            t = labels_batch_cat_[k].reshape([num_out,1]).copy()
            dA[-1] = np.multiply(-t,(np.multiply(A[-1],t))<1)
            dZ[-1] = dA[-1].copy()
            
            
            for i in range(num_h_layer + 1):
                dW[-(i+1)] += np.matmul(dZ[-(i+1)],(A[-(i+2)].T))
                dB[-(i+1)] += dZ[-(i+1)].copy()
                dA[-(i+2)] = np.matmul(W[-(i+1)].T, dZ[-(i+1)])
                dZ[-(i+2)] = np.multiply(dA[-(i+2)],(Z[-(i+2)] > 0))
    
        # Update Parametes
        for i in range(num_h_layer + 1):
            W[i] -= dW[i] * lr
            B[i] -= dB[i] * lr
            
        for i in range(num_h_layer + 1):
            dW[i] = np.zeros([structure[i+1],structure[i]])
            dB[i] = np.zeros([structure[i+1],1])
            
        # Calculating error
        L[0][epoch] = L[0][epoch] + hinge_loss(labels_batch_cat_[k], A[-1])/num_data
    print(epoch, time.time()-t_0)
plt.plot(L[0])

#%% Saving Net
class Net:
    pass

net = Net()
net.W = W
net.B = B
net.num_h_layer = num_h_layer
net.num_in = num_in
net.num_out = num_out
net.structure = structure

with open('MLP_NoPCA.pickle', 'wb') as f:
    pickle.dump([net], f)

#%% Predicting
def predict(data,labels,net):
    S = 0
    num_data = len(data)
    A, Z = [], []
    for i in range(net.num_h_layer + 2):
        A.append(np.zeros([net.structure[i],1]))
        Z.append(np.zeros([net.structure[i],1]))
        
    for k in range(len(data)):
        x = data[k].reshape(im_size,1)
        Z[0] = x #.copy() ?
        A[0] = x
        for i in range(net.num_h_layer + 1):
            Z[i+1] = np.matmul(net.W[i],A[i]) + net.B[i]
            A[i+1] = np.multiply(Z[i+1],(Z[i+1]>0)) #relu function
        A[-1] = Z[-1]
        D = A[-1]
        if np.argmax(D) == np.argmax(labels[k]):
            S +=1
    S = S/num_data
    return S

#%% Testing
x_val = data_val[0][0:10000]
t_val = data_val[1][0:10000]
t_val_cat = to_categorical(t_val)

S = predict(x_val,t_val_cat,net)
print(S)

#%% Run Time Evaluation
time_2 = time.time()
print(f'Loading data :\t{time_1 - time_0}s')
print(f'Total time :\t {time.time() - time_0}s')
print(time.time() - time_0)