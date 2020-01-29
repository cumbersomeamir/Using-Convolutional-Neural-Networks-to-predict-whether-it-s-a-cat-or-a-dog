import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "C:\\Users\\Amir\Desktop\\All programs\\cat-and-dog\\training_set"
CATEGORIES = ["dogs","cats"]

for category in CATEGORIES:
    path =os.path.join(DATADIR, category) #path to cats or dogs Dir
    for img in os.listdir(path):
        #converting image to array
        #reading using os.path.join
        #converting to grey scale as well
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        #DISPLAY image
        plt.imshow(img_array, cmap = "gray")
        plt.show()
        break
    break

print(img_array)
print("Expecting a 2d array because its grayscale")
print(img_array.shape)


IMG_SIZE =50
#Making every image 50 by 50
#display the image and choose your own image size
new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
plt.imshow(new_array,cmap  ='gray')
plt.show()

IMG_SIZE =50
#Making every image 50 by 50
#Creating the training dataset
training_data = []
def create_training_data(): #Iterate and buid the dataset
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        #print(class_num)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                #print(img_array)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
                #print(training_data)
            except Exception as e:
                pass

create_training_data()
print(len(training_data))
print("try to keep the training data balanced , i mean equal number in each category")
#print(training_data)


#Shuffling the data 
import random
random.shuffle(training_data)


#checking wheteher labels are correct or not
for sample in training_data:
    print(sample[1])

#Pack it into the variables
X = [] #feature set
y= []#labels

for features , label in training_data:
    X.append(features)
    y.append(label)
#its ok if y remians a list 
#X needs to be a numpy array
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#the last 1(grayscale)  should be changed to 3 for rgb images and the first -1 means everything
print(X.size)
print(len(y))

#SAVE THE DATA
import pickle
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()











