# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 23:39:36 2020

@author: sinmo
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

def read_data():
    Train = pd.read_csv('train.csv')
    Test = pd.read_csv('test.csv')
    G_data = pd.read_csv('gender_submission.csv')
    return Train, Test, G_data

def polish_data(Data,isTrain=True):
    #Remove the names
    Data= Data.drop(columns=['Name'])
    
    #Normalize the Pclass
    Data['Pclass'] = Data['Pclass']-2
    
    #Sex: label to numbers
    le=preprocessing.LabelEncoder()
    le.fit(['male','female'])
    Data['Sex']=2*le.transform(Data['Sex'])-1
    
    #Remove Ticket number
    Data = Data.drop(columns=['Ticket'])
    
    
    print('Percentage of null Cabin values', Data['Cabin'].isnull().sum(axis = 0)/len(Data['Cabin']))
    print('Will remove this collumn then because of high rate')
    #Remove Cabin
    Data = Data.drop(columns=['Cabin'])    
    
    print('Percentage of null age values', Data['Age'].isnull().sum(axis = 0)/len(Data['Age']))
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #print(Data['Age'][0:20])
    imp.fit(Data['Age'].values.reshape((len(Data['Age']),1)))
    
    Data['Age']=imp.transform(Data['Age'].values.reshape((len(Data['Age']),1)))
    #print(Data['Age'][0:20])
    #print('We need to replace NaN values')
    
    
    #Handle Emabark
    #Null problem
    #print(Train['Embarked'].head())
    print('S  Q  C: ',np.sum(Data['Embarked'].eq('S'))/len(Data['Embarked']),
          np.sum(Data['Embarked'].eq('Q'))/len(Data['Embarked']),
          np.sum(Data['Embarked'].eq('C'))/len(Data['Embarked']))
    Data['Embarked'][Data['Embarked'].isnull()]='S'    
    #Change labels to numbers
    le=preprocessing.LabelEncoder()
    le.fit(['S','Q', 'C'])
    Data['Embarked']=le.transform(Data['Embarked'])-1
    #print(Train['Embarked'].head())
    
    print(Data.keys())
    
    ss=preprocessing.StandardScaler()
    Data=ss.fit_transform(Data)    
    return Data
    
train, test, g_data = read_data()
polish_data(train)
#print(train.keys())
#print(train['Sex'].head())
#print(test.head())
#print(g_data.head())