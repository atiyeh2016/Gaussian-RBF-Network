#%% Importing Libraries
import numpy as np
import gzip, pickle #To load MNIST dataset
import time #To measure time
import matplotlib.pyplot as plt #To plot results
from keras.utils import to_categorical

time_start = time.time()

#%% Defining Hyperparameters
num_data = 50000
lr = 1e-3
epochs = 300
batch = 128
structure = [128,150,10]

#%% Seed
np.random.seed(1234567)

#%% Loading MNIST and Preparing Data
time_start_loadingdata = time.time()

with gzip.open('mnist.pkl.gz','rb') as f :
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data_train, data_val, data_test = u.load()

time_end_loadingdata = time.time()

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

#%% PCA 
def pca(X, n_pca=128):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
  #  assert np.allclose(X.mean(axis=0), np.zeros(m), rtol=1e-6)
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    
    idx = np.argsort(eigen_vals)
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[idx]
    
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs[:,0:128])
    print(X.shape)
    print(eigen_vecs.shape)
    print(X_pca.shape)
    return X_pca, eigen_vals, eigen_vecs

time_pca_start = time.time()
data_pca, eigen_vals, eigen_vecs = pca(data)
time_pca_end = time.time()

#%% Initialization
num_in = structure[0]
num_out = structure[-1]
num_h_layer = len(structure) - 2 #number of hidden layers

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
    
#%% Loss Function
def hinge_loss(T, Y):
    Y = Y.reshape(10,1)
    T = T.reshape(10,1)
    return sum(np.multiply((np.multiply(T,Y)<1),1-np.multiply(Y,T)))

#%% Training
L = np.zeros([1,epochs])
time_train_start = time.time()
for epoch in range(epochs):
    lim1 = 0
    lim2 = batch
    t_0 = time.time()
    while lim2 < num_data:
        data_pca_batch = data_pca[lim1:lim2]
        labels_batch_cat_=labels_cat_[lim1:lim2]
        labels_batch_cat = labels_cat[lim1:lim2]
        lim1 += batch
        lim2 += batch
        
        for k in range(len(data_pca_batch)):
            # Forward:
            x = data_pca_batch[k].reshape(128,1)#.astype('float64')
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
time_train_end = time.time()

plt.plot(L[0], linewidth = 3)
plt.xlabel('epoch')
plt.ylabel('Hinge Loss')
plt.grid()

#%% Saving Net
class Net:
    pass

net = Net()
net.W = W
net.B = B
net.eigen_vals = eigen_vals
net.eigen_vecs = eigen_vecs
net.num_h_layer = num_h_layer
net.num_in = num_in
net.num_out = num_out
net.structure = structure

with open('MLP_PCA.pickle', 'wb') as f:
    pickle.dump([net], f)

#%% Predicting
def predict(data,labels,net):
    S = 0
    data_pca = np.dot(data, net.eigen_vecs[:,0:128])
    num_data = len(data_pca)
    A, Z = [], []
    for i in range(net.num_h_layer + 2):
        A.append(np.zeros([net.structure[i],1]))
        Z.append(np.zeros([net.structure[i],1]))
        
    for k in range(len(data_pca)):
        x = data_pca[k].reshape(128,1)
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

time_test_start = time.time()
S = predict(x_val,t_val_cat,net)
time_test_end = time.time()
print(S)

time_end = time.time()

#%% Run Time Evaluation
print(f'Loading data:\t{time_end_loadingdata - time_start_loadingdata} seconds')
print(f'PCA:          \t{time_pca_end - time_pca_start} seconds')
print(f'Training:\t {time_train_end - time_train_start} seconds')
print(f'Testing:\t {time_test_end - time_test_start} seconds')
print(f'Total:      \t {time_end - time_start} seconds')