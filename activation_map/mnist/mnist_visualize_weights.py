#https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import glob, os
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
load = True
train = False

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_test_inds = y_test.copy()
y_train_inds = y_train.copy()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name='conv1'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu', name='dense1'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', name='dense2'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

if (load):
	print("Loading model...")
	os.chdir(os.path.dirname(__file__))
	model.load_weights('mnist_28x28_epoch12_percent99.08.h5')

if (train):
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Weights of the 1st layer
weights = model.layers[0].get_weights()[0]
print(model.layers[0].name)
#plt.title('Weights of the 1st convolutional layer')
for i in range(32):
        plt.subplot(6, 6, i+1)
        plt.imshow(weights[:,:,0,i], cmap='gray', interpolation='none'); plt.axis('off');

plt.show()

#choose a random data from test set and show probabilities for each class.
'''i = np.random.randint(0,len(x_test))
digit = x_test[i].reshape(28,28)

plt.figure();
plt.subplot(1,2,1);
plt.title('Example of digit: {}'.format(y_test_inds[i]));
plt.imshow(digit,cmap='gray');
plt.axis('off');
probs = model.predict_proba(digit.reshape(1,28,28,1),batch_size=1)
plt.subplot(1,2,2);
plt.title('Probabilities for each digit class');
plt.bar(np.arange(10),probs.reshape(10),align='center'); plt.xticks(np.arange(10),np.arange(10).astype(str));
plt.show()'''
