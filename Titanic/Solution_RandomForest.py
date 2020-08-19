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


from sklearn.ensemble import RandomForestClassifier

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

    Data = pd.get_dummies(Data, columns = ["Sex","Embarked"],
                             prefix=["Sex","Em_type"])
    
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



# Random Forests
random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
random_forest.fit(X_train, Y_train)
out = random_forest.predict(X_test)

out=np.argmax(out,axis=1)
d={'PassengerId':id_test,'Survived':out}
pd.DataFrame(d).to_csv('out_RF.csv',index=None)
print(out)
