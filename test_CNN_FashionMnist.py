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

from keras.models import load_model

#Load model
fashion_model = load_model("fashion_model.h5py")

#Evaluate result
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print("Test loss : ", test_eval[0])
print("Test Accuracy : ", test_eval[1])

predicted_classes = fashion_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

correct = np.where(predicted_classes==test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()
plt.show()

incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()
plt.show()