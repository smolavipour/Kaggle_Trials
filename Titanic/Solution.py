# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 23:39:36 2020

@author: sinmo
"""
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import tensorflow as tf

from sklearn.model_selection import KFold

def read_data():
    Train = pd.read_csv('train.csv')
    Test = pd.read_csv('test.csv')
    G_data = pd.read_csv('gender_submission.csv')
    return Train, Test, G_data

def polish_data(Data):
    #print(Data.isnull().sum())
    #Remove the names
    Data= Data.drop(columns=['Name'])
    
    #Normalize the Pclass
    Data['Pclass'] = Data['Pclass']-2
    
    #Sex: label to numbers    
    Data['Sex']=pd.Categorical(Data['Sex'])
    Data['Sex']=Data.Sex.cat.codes
    Data['Sex']=2*Data['Sex']-1
    
    #Remove Ticket number
    Data = Data.drop(columns=['Ticket'])
    
    
    #Remove Cabin
    Data = Data.drop(columns=['Cabin'])    
    

    Data['Age'].fillna(Data['Age'].mean(), inplace = True)
    Data['Fare'].fillna(Data['Fare'].median(), inplace = True)
    Data['Embarked'].fillna(Data['Embarked'].mode()[0], inplace = True)

    
    #Handle Emabark
    #Change labels to numbers
    Data['Embarked']=pd.Categorical(Data['Embarked'])
    Data['Embarked']=Data.Embarked.cat.codes
    Data['Embarked']=Data['Embarked']-1

    
    return Data
    
train, test, g_data = read_data()
train=polish_data(train)
test=polish_data(test)


X_train=train[train.keys()[2:-1]]
Y_train=train['Survived']
Y_train = to_categorical(Y_train, 2)

X_test=test[test.keys()[1:-1]]
id_test=test['PassengerId']

Feature_Size=len(X_train.keys())

ss=preprocessing.StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)   


#Model Hyperparams
#params

Model_pool=pd.DataFrame([(0.01,200,256),
                         (0.001,150,256),
                         (0.005,200,64),
                         (0.001,100,64),
                         (0.001,100,128)],
                        columns=['LR','Batch_Size','Hidden_Size'])

num_models=len(Model_pool)

# Define the K-fold Cross Validator
num_folds=5
kfold = KFold(n_splits=num_folds, shuffle=True)

Epoch=100

Mean_acc=[]
for i in range(num_models): 
    print('Model #',i)
    LR=Model_pool['LR'][i]
    B_Size=Model_pool['Batch_Size'][i]
    HiddenSize=Model_pool['Hidden_Size'][i]
    model_acc=[]
    
    for ind_train, ind_eval in kfold.split(X_train, Y_train):        
        # Create the model
        model = Sequential()
        model.add(Dense(HiddenSize, input_shape=(Feature_Size,), activation='relu'))
        #model.add(Dense(HiddenSize, input_shape=(Feature_Size,), activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        
        

        #X_test=ss.transform(X_test)   
        
        #Data_train=tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))
        
        # simple early stopping
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
        
        #Optimizer
        opt=keras.optimizers.Adam(learning_rate=LR)
        w0=sum(np.where(Y_train[ind_train]==[1,0])[1])/len(Y_train[ind_train])
        w1=1-w0
        class_weights={0:w0, 1:w1}
        print('weights:',w0,' - ',w1)
        # Configure the model and start training
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(X_train[ind_train],
                  Y_train[ind_train],
                  epochs=Epoch,
                  validation_data=(X_train[ind_eval], Y_train[ind_eval]),
                  batch_size=B_Size,
                  verbose=0,
                  class_weight=class_weights
                  #callbacks=[es]
                  )
        # loss, accuracy
        scores = model.evaluate(X_train[ind_eval], Y_train[ind_eval], verbose=0)
        model_acc.append(scores)
    Mean_acc.append(np.mean(model_acc,axis=0)[1])
    print(Mean_acc[-1],'\n'*2)

Best_model=np.argmax(Mean_acc)    
LR=Model_pool['LR'][Best_model]
B_Size=Model_pool['Batch_Size'][Best_model]
HiddenSize=Model_pool['Hidden_Size'][Best_model]

# Create the model
model = Sequential()
model.add(Dense(HiddenSize, input_shape=(Feature_Size,), activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(Dense(2, activation='softmax'))


#Data_train=tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))

#Optimizer
opt=keras.optimizers.Adam(learning_rate=LR)

# simple early stopping
#es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=30)

w0=sum(np.where(Y_train==[1,0])[1])/len(Y_train)
w1=1-w0
class_weights={0:w0, 1:w1}
print('weights:',w0,' - ',w1)


# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=Epoch, batch_size=B_Size, verbose=1,class_weight=class_weights)


out=model.predict(X_test)
out=np.argmax(out,axis=1)
d={'PassengerId':id_test,'Survived':out}
pd.DataFrame(d).to_csv('out.csv',index=None)
print(out)
