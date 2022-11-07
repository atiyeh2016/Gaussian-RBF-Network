#%% Importing Libraries
import numpy as np
import gzip, pickle #To load MNIST dataset
from keras.utils import to_categorical

#%% Defining Hyperparameters
eps = 0.7
num_data = 20
l = 2
c = 10
num_load_data = 700
im_size = 28*28

#%% Seed
#np.random.seed(1)

#%% Loading MNIST and Preparing Data
with gzip.open('mnist.pkl.gz','rb') as f :
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data_train, data_val, data_test = u.load()

#Seperating training data to data and labels
data = data_test[0]
labels = data_test[1]

data = data[0:num_load_data]
labels = labels[0:num_data]

#Plot the first image of MNIST dataset
#plt.imshow(data[0].reshape(28,28))

labels_cat = to_categorical(labels)
labels_cat_ = labels_cat.copy()
labels_cat_[labels_cat_==0] = -1

#%% Loading Network
class Net:
    pass
with open('Q4_MLP_NoPCA_50000_300_88.pickle', 'rb') as f:
    net = pickle.load(f)
net = net[0]
net.l = l
net.c = c

#%% Initialization
dA, dZ, dW, dB = [], [], [], []
for i in range(net.num_h_layer + 1):
    dW.append(np.zeros([net.structure[i+1],net.structure[i]]))
    dB.append(np.zeros([net.structure[i+1],1]))

for i in range(net.num_h_layer + 2):
    dA.append(np.zeros([net.structure[i],1]))
    dZ.append(np.zeros([net.structure[i],1]))

dW_G = np.zeros([net.l*net.c, net.num_out])
dB_G = np.zeros([net.l*net.c, 1])

#%% Loss Function
def RBF_loss(D,T):
    a1 = np.sum(np.multiply(D,T))
    a2 = np.sum(np.multiply(np.multiply((net.lam-D),(net.lam-D>0)),1-T))
    return a1 + a2

#%% FGSM Attack
def AddingPurturbatoin(data,labels,net):
    A, Z = [], []
    for i in range(net.num_h_layer + 2):
        A.append(np.zeros([net.structure[i],1]))
        Z.append(np.zeros([net.structure[i],1]))
    
    # Forward
    x = data.reshape(im_size,1)
    Z[0] = x #.copy() ?
    A[0] = x
    for i in range(net.num_h_layer + 1):
        Z[i+1] = np.matmul(net.W[i],A[i]) + net.B[i]
        A[i+1] = np.multiply(Z[i+1],(Z[i+1]>0)) #relu function
    A[-1] = Z[-1]

    
    # Backward
    t = labels
    dA[-1] = np.multiply(-t,(np.multiply(A[-1],t))<1)
    dZ[-1] = dA[-1].copy()
     #        dZ[-1] = np.multiply(dA[-1],(Z[-1] > 0))
    for i in range(net.num_h_layer + 1):
        dW[-(i+1)] += np.matmul(dZ[-(i+1)],(A[-(i+2)].T))
        dB[-(i+1)] += dZ[-(i+1)].copy()
        dA[-(i+2)] = np.matmul(net.W[-(i+1)].T, dZ[-(i+1)])
        dZ[-(i+2)] = np.multiply(dA[-(i+2)],(Z[-(i+2)] > 0))
    
    return (x + eps*np.sign(dZ[0]))

#%% Predicting
def predict(data,net):
    A, Z = [], []
    for i in range(net.num_h_layer + 2):
        A.append(np.zeros([net.structure[i],1]))
        Z.append(np.zeros([net.structure[i],1]))
        
    x = data.reshape(im_size,1)
    Z[0] = x #.copy() ?
    A[0] = x
    for i in range(net.num_h_layer + 1):
        Z[i+1] = np.matmul(net.W[i],A[i]) + net.B[i]
        A[i+1] = np.multiply(Z[i+1],(Z[i+1]>0)) #relu function
    A[-1] = Z[-1]
    D = A[-1]
    return np.argmax(D)

#%% Testing
S = 0
for i in range(num_data):
    x_clean = data[i]
    t = labels[i]
    x_noisy = AddingPurturbatoin(x_clean,t,net)
    if t == predict(x_clean,net) & predict(x_clean,net) != predict(x_noisy,net):
        S += 1
    #    print("Real Value: " + str(t))
#    print("Predicted_Clean: " + str(predict(x_clean,net)))
#    print("Predicted_Noisy: " + str(predict(x_noisy,net)))
#    print('----')
print("Attack success rate: "+ str(S/num_data))

