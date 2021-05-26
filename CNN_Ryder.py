#!/usr/bin/env python
# coding: utf-8

# ## CPE 4903 - Cats and Dogs ##

# ## Keras CNN
# 

# In[92]:


import os, cv2, itertools # cv2 -- OpenCV
import numpy as np 
import pandas as pd 
import time
import numpy as np
 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import tensorflow as tf


# In[93]:


TRAIN_DIR = r"C:\Users\Hakeem\Downloads\Ryder"
os.listdir(TRAIN_DIR)[0]
ROWS = 64
COLS = 64
CHANNELS = 3

train_images = [TRAIN_DIR+'\\'+i for i in os.listdir(TRAIN_DIR)]
train_images[0:2]

print(len(train_images))


# In[94]:


for i,image_file in enumerate(train_images) :
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(img_resized)
    plt.show()
    print('i = ', i)
    print(image_file)
    print('Shape of resized image is {}'. format(img_resized.shape))
    time.sleep(2)
    if i==2:
       break
        
print('done')


# In[95]:


img_resized.shape
# falttens matrix to vector rank 0 type
x = np.squeeze(img_resized.reshape((ROWS*COLS*CHANNELS,1)))
x.shape


# In[110]:


def dogCat(image):
    if 'cat' in image_file.lower():
    #print('cat, output = 1')
        return 1
    else:
    #print('dog, output = 0')
        return 0
        


# In[111]:


def read_image(file_path):
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  return cv2.resize(img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC)


# In[146]:


X = np.zeros((ROWS*COLS*CHANNELS, 56))
Y = np.zeros((56))

for i,image_file in enumerate(train_images):
    image = read_image(image_file)
    X[:,i] = np.squeeze(image.reshape((ROWS*COLS*CHANNELS,1)))
    Y[i] = dogCat(image_file)
    
print('done')
print(Y)


# In[113]:


X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.reshape(56,1), test_size = 0.2, random_state = 7)
print('done')


# In[114]:


X_train = (X_train.T)/255
X_test = (X_test.T)/255
Y_train = Y_train.T
Y_test = Y_test.T
m_train = len(Y_train.T)
m_test = len(Y_test.T)

print('X_train shape: ',X_train.shape)
print('X_test shape: ',X_test.shape)
print('Y_train shape: ',Y_train.shape)
print('Y_test shape: ',Y_test.shape)
print(m_train)
print(X_train[0:5])
print(Y_train[0:5])


# In[115]:


X_train2 = X_train.T.reshape(X_train.T.shape[0], 64, 64, 3)
X_test2 = X_test.T.reshape(X_test.T.shape[0], 64, 64, 3)
print('X_train2 shape: ', X_train2.shape)
print('X_test2 shape: ', X_test2.shape)
y_train2 = Y_train.reshape(-1,1)
y_test2 = Y_test.reshape(-1,1)
print('y_train shape: ', y_train2.shape)
print('y_test shape: ', y_test2.shape)


# In[116]:


model_cnn = Sequential()

model_cnn.add(Conv2D(64,(3, 3), activation = 'relu'))
model_cnn.add(MaxPooling2D(pool_size = (2,2)))
model_cnn.add(Dropout(0.2))

model_cnn.add(Conv2D(128,(3, 3), activation = 'relu'))
model_cnn.add(MaxPooling2D(pool_size = (2,2)))
model_cnn.add(Dropout(0.2))

model_cnn.add(Flatten())

model_cnn.add(Dense(units = 200, activation = 'relu'))
model_cnn.add(Dense(units = 1, activation = 'sigmoid'))


# In[117]:


model_cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[147]:


history = model_cnn.fit(X_train2,Y_train.T,epochs=10,batch_size=5,validation_split=.2,verbose=1)
print(model_cnn.summary())


# In[148]:


correct = 0
wrong = 0

size = len(Y_test[0])
Yhat = model_cnn.predict(X_test2);

for i in range(size):
    if(Y_test.T[i] != np.rint(Yhat[i])):
        wrong = wrong + 1;
correct = 5000 -  wrong;

train_acc = (history.history['accuracy'][-1]* 100);
test_acc = (correct/size) * 100;

print("The train accuracy is: ", np.rint(train_acc))
print("The test accuracy is: ", np.rint(test_acc))

plt.plot(history.history['loss']);


# In[157]:


file = r"C:\Users\Hakeem\Downloads\train\cat.1.jpg"
test_img = read_image(file)
fig, (ax1) = plt.subplots(1)
ax1.imshow(test_img)
plt.show()

#test_img = np.squeeze(test_img.reshape((ROWS*COLS*CHANNELS,1)))

test_img = test_img/255
test_img = test_img.reshape(1, 64, 64, 3)
test_img.shape


# In[158]:


result  = model_cnn.predict(test_img)
print(result)


# In[ ]:




