#https://sempwn.github.io/blog/2017/04/06/conv_net_intro
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
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

# convert class vectors to binary class matrices
y_test_inds = y_test.copy()
y_train_inds = y_train.copy()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape = input_shape, kernel_initializer = 'normal', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (5, 5), kernel_initializer = 'normal', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

if (load):
	print("Loading model...")
	os.chdir(os.path.dirname(__file__))
	model.load_weights('mnist_28x28_percent98.96.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Create new sequential model, same as before but just keep the convolutional layer.
model_new = Sequential()
model_new.add(Conv2D(32, (5, 5), input_shape = input_shape, kernel_initializer = 'normal', activation = 'relu'))

#set weights for new model from weights trained on MNIST.
for i in range(1):
	model_new.layers[i].set_weights(model.layers[i].get_weights())

#pick a random digit and "predict" on this digit (output will be first layer of CNN)
#i = np.random.randint(0, len(x_test))
def show_activation(layer):
	digit = x_test[layer].reshape(1, 28, 28, 1)
	pred = model_new.predict(digit)

	#For all the filters, plot the output of the input
	plt.figure(figsize=(18, 18))
	filts = pred[0]
	for i in range(32):
			filter_digit = filts[:,:,i]
			plt.subplot(6, 6, i+1)
			plt.imshow(filter_digit, cmap='gray'); plt.axis('off');

	plt.show()

def show_activation_values(layer, filter):
	digit = x_test[layer].reshape(1, 28, 28, 1)
	pred = model_new.predict(digit)
	return pred

show_activation(0)
output = show_activation_values(0, 0)