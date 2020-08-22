import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn import preprocessing

def read_data():
    Train = pd.read_csv('train.csv')
    Test = pd.read_csv('test.csv')
    return Train, Test

train, test= read_data()

X_train=train[train.keys()[1:]]
Y_train=train['label']
Y_train = to_categorical(Y_train, 10)


X_test=test


Feature_Size=784

ss=preprocessing.StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)   



#Model Hyperparams
#params

LR_ = [0.01, 0.001]
Batch_Size_ = [50, 100]
Hidden_Size_ = [ 32, 128]
Epoch_ = [100,200]
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
num_folds=5
kfold = KFold(n_splits=num_folds, shuffle=True)



Mean_acc=[]
for i in range(num_models): 
    print('Model #',i)
    LR=Model_pool['LR'][i]
    B_Size=Model_pool['Batch_Size'][i]
    HiddenSize=Model_pool['Hidden_Size'][i]
    Epoch=Model_pool['Epoch'][i]
    Op=Model_pool['Optimizer'][i]
    model_acc=[]
    
    
    for ind_train, ind_eval in kfold.split(X_train, Y_train):        
        # Create the model
        model = Sequential()
        model.add(Dense(HiddenSize, input_shape=(Feature_Size,), activation='relu'))
        #model.add(Dense(HiddenSize, input_shape=(Feature_Size,), activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(Dense(10, activation='softmax'))
        
        

        #X_test=ss.transform(X_test)   
        
        #Data_train=tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))
        
        # simple early stopping
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
        
        #Optimizer
        if Op=='adam':
            opt=keras.optimizers.Adam(learning_rate=LR)
        elif Op=='sgd':
            opt=keras.optimizers.SGD(learning_rate=LR)
            
        # Configure the model and start training
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(X_train[ind_train],
                  Y_train[ind_train],
                  epochs=Epoch,
                  validation_data=(X_train[ind_eval], Y_train[ind_eval]),
                  batch_size=B_Size,
                  verbose=1
                  #callbacks=[es]
                  )
        
        
        # loss, accuracy
        scores = model.evaluate(X_train[ind_eval], Y_train[ind_eval], verbose=0)
        model_acc.append(scores)
        print('Fold accuracy: ',model_acc[-1][1])
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
model.add(Dense(10, activation='softmax'))


#Data_train=tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))

#Optimizer
if Model_pool['Optimizer'][Best_model]=='adam':
    opt=keras.optimizers.Adam(learning_rate=LR)
elif  Model_pool['Optimizer'][Best_model]=='sgd':
    opt=keras.optimizers.SGD(learning_rate=LR)

# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=Epoch, batch_size=B_Size, verbose=1)


out=model.predict(X_test)
out=np.argmax(out,axis=1)
id_test=np.arange(0,28000)+1
d={'ImageId':id_test,'Label':out}
pd.DataFrame(d).to_csv('out_MLP.csv',index=None)
