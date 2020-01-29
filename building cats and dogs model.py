# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 04:45:54 2019

@author: Amir
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten , Conv2D , MaxPooling2D

import pickle 
X= pickle.load(open("X.pickle", "rb"))
y= pickle.load(open("y.pickle", "rb"))

#Before feeding it to the neural network we need to normalize the data
#We know the min and max of each array (0,255) for pixel data 

X = X/255.0
#print(X[1:5])
print(len(X))
#print(y[1:5])

print(len(y))


model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))


#Output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer ="adam",metrics= ['accuracy'])

model.fit(X, y , batch_size=20,epochs=20 ,validation_split= 0.1)

