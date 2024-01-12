#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:26:16 2023

@author: songyuting
"""

import keras
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam, Nadam, Adamax
import time
import numpy as np


(x_train, y_train), (x_test, y_test)= mnist.load_data()

x_train= x_train.reshape(60000, 784)
x_test= x_test.reshape(10000,784)
x_train= x_train.astype('float32')
x_test= x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#%%

batch_size= 128
num_classes= 10
epoch =10

y_train= keras.utils.to_categorical(y_train, num_classes)
y_test= keras.utils.to_categorical(y_test, num_classes)

#%%
model = Sequential()

#%%

model.add(Dense(10, activation='softmax',  input_shape= (784,)))

model.summary()

#%%
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history= model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epoch,
                   verbose=1,
                   validation_data=(x_test, y_test))

#%%
indice = slice(0,20)
mini_test_x= list(x_test[indice])
mini_test_y= list(y_test[indice])
plt.figure()
for idx,(testdata,labeldata) in enumerate(zip(mini_test_x,mini_test_y)):
    plt.subplot(4,5,idx+1)
    testdata = np.expand_dims(testdata, axis=0)
    predict_prob =model.predict(testdata)[0]
    predict_maxval = np.max(predict_prob)
    predict_class = predict_prob.tolist().index(predict_maxval)
    title_obj = plt.title('predict_answer:' +str(predict_class))
    plt.imshow(testdata.reshape(28,28))
    plt.axis("off")

             
    label_prob = labeldata
    label_class=label_prob.tolist().index(1)
    #print('Quest:',idxt1, '\nTrue_Answer: ',label_class, '\nModel_predict:',predict_class,'\n')
    if label_class != predict_class:
        plt.setp(title_obj,color='r')



plt.show()

