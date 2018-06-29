from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import glob, os
from scipy.misc import imsave
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 5

img_size = 28
load = True
save = False
train = False

# Loading data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_size, img_size, 1)
x_test = x_test.reshape(x_test.shape[0], img_size, img_size, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CNN model
input_shape = (img_size, img_size, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name='conv_1'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv_2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu', name='dense_1'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', name='output'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#model.summary()
if (load):
	print("Loading model...")
	os.chdir(os.path.dirname(__file__))
	model.load_weights('mnist_28x28_epoch12_percent99.08.h5')

if (train):
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

if (save):
	print("Saving model...")
	os.chdir(os.path.dirname(__file__))
	model.save_weights('mnist_' + str(img_size) + 'x' + str(img_size) + '.h5')

def layer_to_visualize(layer, img):
	inputs = [K.learning_phase()] + model.inputs

	_convout1_f = K.function(inputs, [layer.output])
	def convout1_f(X):
		# The [0] is to disable the training phase flag
		return _convout1_f([0] + [X])

	convolutions = convout1_f(img)
	convolutions = np.squeeze(convolutions)

	print ('Shape of conv:', convolutions.shape)

	n = convolutions.shape[0]
	n = int(np.ceil(np.sqrt(n)))

	# Visualization of each filter of the layer
	fig = plt.figure(figsize=(12,8))
	for i in range(len(convolutions)):
		ax = fig.add_subplot(n, n, i+1)
		ax.imshow(convolutions[i], cmap='gray')
	plt.show()

layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_img_data = np.expand_dims(x_test[65], axis=0)
layer_to_visualize(layer_dict['conv_1'], input_img_data)

'''print('Layer name: ' + model.layers[0].name)
l1_weights = model.layers[0].get_weights()[0]

# Activation maps
step = 1.
input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = 'conv_1'

def get_activation_map(filter_index):
	layer_output = layer_dict[layer_name].output
	loss = K.mean(layer_output[:, :, :, filter_index])

	grads = K.gradients(loss, input_img)[0]
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	iterate = K.function([input_img], [loss, grads])

	input_img_data = np.random.random((1, img_size, img_size, 1)) * 20 + 128.
	#input_img_data = x_test[11]
	#input_img_data = np.expand_dims(input_img_data, axis=0)
	for i in range(250):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step

	return input_img_data

def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	x += 0.5
	x = np.clip(x, 0, 1)

	x *= 255
	#x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def show_grayscale_image(img_data):
	cols, rows = 5, 5
	fig = plt.figure(figsize=(5, 5))
	for i in range(1, cols*rows +1):
		fig.add_subplot(rows, cols, i)
		plt.imshow(np.reshape(img_data[i-1], (img_size, img_size)).astype(np.uint8), cmap=plt.cm.gray)
	#plt.imshow(np.reshape(img_data, (img_size, img_size)).astype(np.uint8), cmap=plt.cm.gray)
	plt.show()

img = []
for k in range(0, 25):
	img.append(deprocess_image(get_activation_map(k)))

img = np.reshape(img, (25, 28, 28, 1))
#show_grayscale_image(img)
show_grayscale_image(img)'''