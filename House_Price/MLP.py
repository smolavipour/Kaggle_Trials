import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.utils import to_categorical

#import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn import preprocessing

   
Train_raw = pd.read_csv('Data/train.csv')
Test_raw = pd.read_csv('Data/test.csv')

def prepare_data(train, test):      
    # Remove bad features
    for k in train.keys():
        if train[k].isnull().sum() >10:
            train = train.drop(columns=[k])
    
    # Select most relevant features (top 12)
    crr = train[list(train.keys())].corr()
    crr_sort = crr['SalePrice'].sort_values(ascending=False)
    key_list = list(crr_sort[0:13].keys()) 
    key_list.append('Neighborhood')
    

    X_train = train[key_list]
    X_train = X_train.drop(columns=['SalePrice'])
    X_test = test[X_train.keys()]

    X_data = pd.concat([X_train, X_test], ignore_index= True)
    
    # Handle NA
    for feature in X_data.keys():
        X_data[feature] = X_data[feature].fillna(X_data[feature].mode()[0])    
     
    # Handle Skewed data
    for k in X_data.keys():
        if X_data[k].dtype!=object:
            if abs(X_data[k].skew())>1:
               X_data[k] = X_data[k].map(lambda i: np.log1p(i)) 
    
    # Add a new feature
    X_data['total_sf'] =  X_data['TotalBsmtSF'] + X_data['1stFlrSF']
    
    neighbors = train['Neighborhood'].unique()
    X_data['Neighborhood_Qual']=-1
    for ne in neighbors:
        if train['SalePrice'].loc[train['Neighborhood'] == ne].mean()>180000:
            X_data['Neighborhood_Qual'].loc[X_data['Neighborhood'] == ne] = 1    
    X_data = X_data.drop(columns=['Neighborhood'])
        
    #Change labels to numbers
    #Cat_f = ['MSSubClass','Neighborhood','BldgType']
    Cat_f = []
    for k in Cat_f:
        X_data[k] = pd.Categorical(X_data[k])
        X_data[k] = X_data[k].cat.codes
    
    X_data = pd.get_dummies(X_data, columns = Cat_f, prefix= Cat_f)
    
    X_train = X_data.iloc[:len(train)]
    X_test = X_data.iloc[len(train):]
    return X_train, X_test

X_Train, X_Test = prepare_data(Train_raw, Test_raw)
Y_Train = Train_raw['SalePrice']    

ss=preprocessing.StandardScaler()
X_Train=ss.fit_transform(X_Train)
X_Test=ss.transform(X_Test) 

X_Train = pd.DataFrame(X_Train)
X_Test = pd.DataFrame(X_Test)

# Model params
#Feature_Size = len(X_Train.keys())
Feature_Size = X_Train.shape[1]
LR_ = [0.01, 0.1]
Batch_Size_ = [100]
Hidden_Size_ = [32, 128]
Epoch_ = [400, 600]
Optimizer_=['adam']

Model_pool=pd.DataFrame(columns=['LR','Batch_Size','Hidden_Size', 'Epoch', 'Optimizer'])

for lr in LR_:
    for bs in Batch_Size_:
        for hs in Hidden_Size_:
            for ep in Epoch_:
                for op in Optimizer_:
                    Model_pool=Model_pool.append({'LR':lr,
                                                  'Batch_Size':bs,
                                                  'Hidden_Size':hs,
                                                  'Epoch':ep,
                                                  'Optimizer':op},ignore_index=True)

num_models=len(Model_pool)

# Define the K-fold Cross Validator
num_folds=4
kfold = KFold(n_splits=num_folds, shuffle=True)



Mean_loss=[]
for i in range(num_models): 
    print('Model #',i)
    LR=Model_pool['LR'][i]
    B_Size=Model_pool['Batch_Size'][i]
    HiddenSize=Model_pool['Hidden_Size'][i]
    Epoch=Model_pool['Epoch'][i]
    Op=Model_pool['Optimizer'][i]
    model_loss=[]
    
    for ind_train, ind_eval in kfold.split(X_Train, Y_Train):        
        # Create the model
        model = Sequential()
        model.add(Dense(HiddenSize, input_shape=(Feature_Size,), activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(Dense(1))
        
        #Optimizer
        if Op=='adam':
            opt=keras.optimizers.Adam(learning_rate=LR)
        elif Op=='sgd':
            opt=keras.optimizers.SGD(learning_rate=LR)
            
        # Configure the model and start training
        model.compile(loss='mean_squared_error', optimizer=opt)
        model.fit(X_Train.iloc[ind_train],
                  Y_Train.iloc[ind_train],
                  epochs=Epoch,
                  validation_data=(X_Train.iloc[ind_eval], Y_Train.iloc[ind_eval]),
                  batch_size=B_Size,
                  verbose=0
                  )
        # loss, accuracy
        scores = model.evaluate(X_Train.iloc[ind_eval], Y_Train.iloc[ind_eval], verbose=0)
        model_loss.append(scores)
        print('Fold loss: ', model_loss[-1]/1e6)
    Mean_loss.append(np.mean(model_loss))
    print(Mean_loss[-1]/1e6,'\n'*2)

Best_model=np.argmin(Mean_loss)    
LR=Model_pool['LR'][Best_model]
B_Size=Model_pool['Batch_Size'][Best_model]
HiddenSize=Model_pool['Hidden_Size'][Best_model]

# Create the model
model = Sequential()
model.add(Dense(HiddenSize, input_shape=(Feature_Size,), activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(Dense(1))

opt=keras.optimizers.Adam(learning_rate=LR)


model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
model.fit(X_Train, Y_Train, epochs=Epoch,
          validation_data=(X_Train, Y_Train),
          batch_size=B_Size,
          verbose=0
          )

#out = model.predict(X_Train)
#print(np.linalg.norm(np.log(out.T[0]) - np.log(np.asarray(Y_Train))) / np.sqrt(len(out)))    
out = model.predict(X_Test)
nan_ids = np.where(np.isnan(out))[0]
out[nan_ids] = np.mean(Y_Train)

d = {'Id':Test_raw['Id'], 'SalePrice':np.round(out.T[0],3)}
pd.DataFrame(d).to_csv('out_MLP3.csv',index=None)

    
        
        




