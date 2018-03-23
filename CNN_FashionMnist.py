# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:34:17 2018

@author: Olan
"""

from keras.datasets import fashion_mnist
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def output_detail():
      classes = np.unique(train_Y)
      nClasses = len(classes)
      print('Total number of outputs : ', nClasses)
      print('Output classes : ', classes)

def plot_input():
      plt.figure(figsize=[5,5])
      
      # Display the first image in training data
      plt.subplot(121)
      plt.imshow(train_X[0,:,:], cmap='gray')
      plt.title("Ground Truth : {}".format(train_Y[0]))
      
      # Display the first image in testing data
      plt.subplot(122)
      plt.imshow(test_X[0,:,:], cmap='gray')
      plt.title("Ground Truth : {}".format(test_Y[0]))
      plt.show()

# plot_input()

train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)     

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# Normalize to be 0 - 1.
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding (0 for not that class 1 for that class - Ex 0 0 0 1 0 0 0)
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

print('Original label : ', train_Y[0])
print('One hot coding laball : ', train_Y_one_hot[0])

# Split train into train and validation
from sklearn.model_selection import train_test_split
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print('train_X : ', train_X.shape)
print('valid_X : ', valid_X.shape)
print('train_label : ', train_label.shape)
print('valid_label : ', valid_label.shape)

#Learning
import keras 
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20
num_classes = 10

#Add network
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.0))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.0))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.0))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.0))                  
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

fashion_model.summary()

#Train network
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
fashion_model.save("fashion_model.h5py")



