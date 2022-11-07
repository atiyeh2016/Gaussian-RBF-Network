#%% Importing Libraries
import numpy as np
import gzip, pickle #To load MNIST dataset
import time #To measure time
import matplotlib.pyplot as plt #To plot results
from keras.utils import to_categorical

time_start = time.time()

#%% Defining Hyperparameters
lr = 1e-4
epochs = 50
batch = 128
momentum = 0
num_data = 10000
n_pca = 128
im_size = 28*28
structure = [128,150,10]
c = 10 # number of classes
l = 2 # RBF dimension
sigma = 0.3
n_noisy_data = 400
num_in = structure[0]
num_out = structure[-1]
num_h_layer = len(structure) - 2 #number of hidden layers
lam = 1500

#%% Seed
np.random.seed(1234567)

#%% Loading MNIST and Preparing Data
time_start_loadingdata = time.time()

#Loading MNIST dataset from file
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
    
    # Project X onto PCA space
    X_pca = np.dot(X, eigen_vecs[:,0:128])
    print(X.shape)
    print(eigen_vecs.shape)
    print(X_pca.shape)
    return X_pca, eigen_vals, eigen_vecs

time_pca_start = time.time()
data_pca, eigen_vals, eigen_vecs = pca(data)
time_pca_end = time.time()

#%% Adding Rejection Class
data_length = n_pca
noisy_data = sigma * np.random.randn(n_noisy_data,data_length)
data_pca = np.concatenate((data_pca, noisy_data), axis=0)
extra_zeros = np.zeros([n_noisy_data,num_out])
labels_cat = np.concatenate((labels_cat,extra_zeros), axis = 0)
cc = list(zip(data_pca, labels_cat))
np.random.shuffle(cc)
data_pca, labels_cat = zip(*cc)
data_pca = np.array(data_pca)
labels_cat = np.array(labels_cat)

#%% Initialization
W, B, A, Z, vel_W, vel_B = [], [], [], [], [], []
dA, dZ, dW, dB = [], [], [], []

#Gaussian RBF Parameters
W_G = np.random.rand(l*c, num_out)/1000
B_G = np.random.rand(l*c, 1)/1000
vel_W_G = np.zeros([l*c, num_out])
vel_B_G = np.zeros([l*c, 1])
A2D = np.repeat(np.eye(c),l,axis=1)

for i in range(num_h_layer + 1):
    W.append(np.random.rand(structure[i+1],structure[i])/1000)
    dW.append(np.zeros([structure[i+1],structure[i]]))
    B.append(np.random.rand(structure[i+1],1)/1000)
    dB.append(np.zeros([structure[i+1],1]))
    vel_W.append(np.random.rand(structure[i+1],structure[i])/1000)
    vel_B.append(np.random.rand(structure[i+1],1)/1000)

dW_G = np.zeros([l*c, num_out])
dB_G = np.zeros([l*c, 1])

for i in range(num_h_layer + 2):
    A.append(np.zeros([structure[i],1]))
    Z.append(np.zeros([structure[i],1]))
    dA.append(np.zeros([structure[i],1]))
    dZ.append(np.zeros([structure[i],1]))

#%% Loss Function
def RBF_loss(D,T):
    a1 = np.sum(np.multiply(D,T))
    a2 = np.sum(np.multiply(np.multiply((lam-D),(lam-D>0)),1-T))
    return a1 + a2

#%% Training
L = np.zeros([1,epochs])
time_train_start = time.time()
for epoch in range(epochs):
    lim1 = 0
    lim2 = batch
    t_0 = time.time()
    while lim2 < num_data:
        data_pca_batch = data_pca[lim1:lim2]
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
            
            # Gaussian RBF
            Z_G = np.matmul(W_G, A[-1]) + B_G
            A_G = np.abs(Z_G)
            D = np.matmul(A2D, A_G)
            
            # Backward
            t_G = labels_batch_cat[k].reshape([num_out,1]).copy()
            if sum(labels_batch_cat[k]) != 0: 
                dd = np.multiply((1-t_G),-1*(lam-D > 0)) + t_G
            else:
                dd = -1*(lam-D > 0)
            dA_G = np.repeat(dd,l,axis=0)
            dZ_G = np.multiply(np.sign(Z_G), dA_G)
            dW_G += (np.matmul(dZ_G, A[-1].T))
            dB_G += (dZ_G.copy())
            dA[-1] = np.matmul(W_G.T, dZ_G)
    #        t = labels_cat_[k].reshape([num_out,1]).copy()
    #        dA[-1] = np.multiply(-t,(np.multiply(A[-1],t))<1)
            dZ[-1] = dA[-1].copy()
    #        dZ[-1] = np.multiply(dA[-1],(Z[-1] > 0))
            for i in range(num_h_layer + 1):
                dW[-(i+1)] += (np.matmul(dZ[-(i+1)],(A[-(i+2)].T)))
                dB[-(i+1)] += (dZ[-(i+1)].copy())
                dA[-(i+2)] = np.matmul(W[-(i+1)].T, dZ[-(i+1)])
                dZ[-(i+2)] = np.multiply(dA[-(i+2)],(Z[-(i+2)] > 0))
    
        # Update Parametes
        for i in range(num_h_layer + 1):
            vel_W[i] = vel_W[i]*momentum - dW[i]*lr
            vel_B[i] = vel_B[i]*momentum - dB[i]*lr
            W[i] += vel_W[i]
            B[i] += vel_B[i]
        vel_W_G = vel_W_G*momentum - dW_G*lr
        vel_B_G = vel_B_G*momentum - dB_G*lr
        W_G += vel_W_G
        B_G += vel_B_G
        
        for i in range(num_h_layer + 1):
            dW[i] = np.zeros([structure[i+1],structure[i]])
            dB[i] = np.zeros([structure[i+1],1])

        dW_G = np.zeros([l*c, num_out])
        dB_G = np.zeros([l*c, 1])
        
        # Calculating error
        L[0][epoch] = L[0][epoch] + RBF_loss(D, t_G)/num_data
    
    print(epoch, time.time()-t_0)
time_train_end = time.time()
plt.plot(L[0])

#%% Saving Net
class Net:
    pass

net = Net()
net.W = W
net.W_G = W_G
net.B = B
net.B_G = B_G
net.eigen_vals = eigen_vals
net.eigen_vecs = eigen_vecs
net.num_h_layer = num_h_layer
net.num_in = num_in
net.num_out = num_out
net.l = l
net.c = c
net.lam = lam
net.A2D = A2D
net.structure = structure

with open('GRBF_Rejection_lam_1500.pickle', 'wb') as f:
    pickle.dump([net], f)

#%% Predicting
def predict(data_pca,labels,net):
    S = 0
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

        Z_G = np.matmul(net.W_G, A[-1]) + net.B_G
        A_G = np.abs(Z_G)
        D = np.matmul(net.A2D, A_G)
        if all(D>lam):
            if not any(labels[k]):
                S += 1
        elif np.argmin(D) == np.argmax(labels[k]):
            S +=1
    S = S/num_data
    return S

#%% Testing
num_val_data = 10000
x_val = data_val[0][0:num_val_data]
x_val = np.dot(x_val, net.eigen_vecs[:,0:128])
t_val = data_val[1][0:num_val_data]
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

