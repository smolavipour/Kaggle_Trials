import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn import preprocessing

def data_preparation(D,isTest=False):
    out=[]
    n=len(D)
    for i in range(n):
        im=D.loc[i,:]
        if not isTest:
            im=im[1:]
        im=im.values.reshape((1,28,28,1))
        
        out.append(im)
    out=np.concatenate(out,axis=0)    
    return out


def read_data():
    Train = pd.read_csv('train.csv')
    Test = pd.read_csv('test.csv')
    return Train, Test

train, test= read_data()

X_train=data_preparation(train)
Y_train=train['label']
Y_train = to_categorical(Y_train, 10)

X_test=data_preparation(test,isTest=1)

print("Train Shape: {}\nTest Shape: {}".format(X_train.shape, X_test.shape))


#Define hyper parameters
epochs = 10
learning_rate = 0.001
batch_size = 128


n_input = 28
n_classes = 10

x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, stride = 1):
    x = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k = 2):
    return tf.nn.max_pool(x, ksize = [1, k , k , 1], strides = [1, k , k, 1], padding = "SAME")




#Predict
#out=model.predict(X_test)
#out=np.argmax(out,axis=1)
#id_test=np.arange(0,28000)+1
#d={'ImageId':id_test,'Label':out}
#pd.DataFrame(d).to_csv('out_MLP.csv',index=None)
