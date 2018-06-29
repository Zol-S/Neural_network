from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model

from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflowjs as tfjs

K.set_learning_phase(1)

# the data, split between train and test sets
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

lb = preprocessing.LabelBinarizer()
lb.fit(y_test)
y_test = lb.transform(y_test)

# Loading model
model = load_model('mnist_28x28_percent98.96.h5')
#score, acc = model.evaluate(x_test, y_test, verbose = 1)
#print('\nLoss:', score, '\nAcc:', acc)

tfjs.converters.save_keras_model(model, 'mnist_28x28_percent98.96.json')
