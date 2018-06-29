# https://www.kaggle.com/ernie55ernie/mnist-with-keras-visualization-and-saliency-map

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
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

'''print('Number of occurence for each number in training data (0 stands for 10):')
print(np.vstack((np.unique(y_train), np.bincount(y_train))).T)

# plot first 36 images in MNIST
fig, ax = plt.subplots(6, 6, figsize = (12, 12))
fig.suptitle('First 36 images in MNIST')
fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
for x, y in [(i, j) for i in range(6) for j in range(6)]:
	ax[x, y].imshow(x_train[x + y * 6].reshape((28, 28)), cmap = 'gray')
	ax[x, y].set_title(y_train[x + y * 6])'''

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

# transform training label to one-hot encoding
lb = preprocessing.LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)

# split training and validating data
print('Stratified shuffling...')
sss = StratifiedShuffleSplit(10, 0.2, random_state = 15)
for train_idx, val_idx in sss.split(x_train, y_train):
	x_train_tmp, x_val = x_train[train_idx], x_train[val_idx]
	y_train_tmp, y_val = y_train[train_idx], y_train[val_idx]

x_train = x_train_tmp
y_train = y_train_tmp
print('Finish stratified shuffling...')

# CNN
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

img_size = (28, 28, 1)
n_classes = 10

if os.path.exists('keras_model.h5'):
	print('Loading model...')
	model = load_model('keras_model.h5')
else:
	print('Building model...')
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape = img_size, kernel_initializer = 'normal'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Conv2D(64, (5, 5), kernel_initializer = 'normal'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(n_classes))
	model.add(Activation('softmax'))

	model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

datagen = ImageDataGenerator(
	featurewise_center = False,
	samplewise_center = False,
	featurewise_std_normalization = False,
	samplewise_std_normalization = False,
	zca_whitening = False,
	rotation_range = 0,
	zoom_range = 0.1,
	width_shift_range = 0.1,
	height_shift_range = 0.1,
	horizontal_flip = False,
	vertical_flip = False
)

datagen.fit(x_train)

print('Training model...')
model.fit_generator(datagen.flow(x_train, y_train, batch_size = 1000),
					epochs = 20,
					validation_data = (x_val, y_val),
					steps_per_epoch = x_train.shape[0] / 1000,
					verbose = 1)
print('Validating model...')
score, acc = model.evaluate(x_val, y_val, verbose = 1)
print('\nLoss:', score, '\nAcc:', acc)
model.save('keras_model.h5')

print('Predicting...')
y_test = model.predict(x_test)
y_test = lb.inverse_transform(y_test)
y_test = [[y] for y in y_test]
index = [[i] for i in range(1, x_test.shape[0] + 1)]
output_np = np.concatenate((index, y_test), axis = 1)