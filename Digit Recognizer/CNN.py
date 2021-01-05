import numpy as np
import pandas as pd


from keras.utils import to_categorical

from tensorflow.keras import layers, models

import matplotlib.pyplot as plt

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


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=epochs, 
                    validation_data=(X_train, Y_train))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

out=model.predict(X_test.astype(np.float64))
out=np.argmax(out,axis=1)
id_test=np.arange(0,28000)+1
d={'ImageId':id_test,'Label':out}
pd.DataFrame(d).to_csv('out_CNN.csv',index=None)

#Predict
#out=model.predict(X_test)
#out=np.argmax(out,axis=1)
#id_test=np.arange(0,28000)+1
#d={'ImageId':id_test,'Label':out}
#pd.DataFrame(d).to_csv('out_MLP.csv',index=None)
