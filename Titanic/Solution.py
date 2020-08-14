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

import tensorflow as tf

def read_data():
    Train = pd.read_csv('train.csv')
    Test = pd.read_csv('test.csv')
    G_data = pd.read_csv('gender_submission.csv')
    return Train, Test, G_data

def polish_data(Data):
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
    
    
    #print('Percentage of null Cabin values', Data['Cabin'].isnull().sum(axis = 0)/len(Data['Cabin']))
    #print('Will remove this collumn then because of high rate')
    #Remove Cabin
    Data = Data.drop(columns=['Cabin'])    
    
    #print('Percentage of null age values', Data['Age'].isnull().sum(axis = 0)/len(Data['Age']))
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #print(Data['Age'][0:20])
    imp.fit(Data['Age'].values.reshape((len(Data['Age']),1)))
    
    Data['Age']=imp.transform(Data['Age'].values.reshape((len(Data['Age']),1)))
    #print(Data['Age'][0:20])
    #print('We need to replace NaN values')
    
    
    #Handle Emabark
    #Null problem
    #print(Train['Embarked'].head())
    #print('S  Q  C: ',np.sum(Data['Embarked'].eq('S'))/len(Data['Embarked']),
    #      np.sum(Data['Embarked'].eq('Q'))/len(Data['Embarked']),
    #      np.sum(Data['Embarked'].eq('C'))/len(Data['Embarked']))
    Data['Embarked'][Data['Embarked'].isnull()]='S'    
    #Change labels to numbers
    Data['Embarked']=pd.Categorical(Data['Embarked'])
    Data['Embarked']=Data.Embarked.cat.codes
    Data['Embarked']=Data['Embarked']-1
    #print(Train['Embarked'].head())    
    
    return Data
    
train, test, g_data = read_data()
train=polish_data(train)
test=polish_data(test)


X_train=train[train.keys()[2:-1]]
Y_train=train['Survived']
Y_train = to_categorical(Y_train, 2)

X_test=test[test.keys()[1:-1]]
id_test=test['PassengerId']

#print(X_train.head())
#print(X_test.head())

Feature_Size=len(X_train.keys())

# Create the model
model = Sequential()
model.add(Dense(256, input_shape=(Feature_Size,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))


ss=preprocessing.StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)   

#Data_train=tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))

# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=200, verbose=1, validation_split=0.2)
out=model.predict(X_test)
out=np.argmax(out,axis=1)
d={'PassengerId':id_test,'Survived':out}
pd.DataFrame(d).to_csv('out.csv')
