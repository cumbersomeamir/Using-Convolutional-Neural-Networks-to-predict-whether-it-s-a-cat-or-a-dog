# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:51:06 2019

@author: Amir
"""

import cv2
import tensorflow as tf

CATEGORIES = ["dogs","cats"]



def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)


#LOADING IN THE MODEL
model = tf.keras.models.load_model("trained cats and dogs model")

#predict always predicts on a list
prediction = model.predict([prepare('its a dog.jpg')])
print(prediction)
print(prediction[0][0])
print(int(prediction[0][0]))
print(CATEGORIES[int(prediction[0][0])])


