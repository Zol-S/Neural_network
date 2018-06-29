#https://sempwn.github.io/blog/2017/04/06/conv_net_intro
import keras

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import glob, os
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
	print('----- activations -----')
	activations = []
	inp = model.input

	model_multi_inputs_cond = True
	if not isinstance(inp, list):
		# only one input! let's wrap it in a list.
		inp = [inp]
		model_multi_inputs_cond = False

	outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]  # all layer outputs
	funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation function

	if model_multi_inputs_cond:
		list_inputs = []
		list_inputs.extend(model_inputs)
		list_inputs.append(0.)
	else:
		list_inputs = [model_inputs, 0.]

	# Learning phase. 0 = Test mode (no dropout or batch normalization)
	# layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
	layer_outputs = [func(list_inputs)[0] for func in funcs]
	for layer_activations in layer_outputs:
		activations.append(layer_activations)
		if print_shape_only:
			print(layer_activations.shape)
		else:
			print(layer_activations)
	return activations

def display_activations(activation_maps):
	import numpy as np
	import matplotlib.pyplot as plt

	batch_size = activation_maps[0].shape[0]
	assert batch_size == 1, 'One image at a time to visualize.'
	for i, activation_map in enumerate(activation_maps):
		print('Displaying activation map {}'.format(i))
		shape = activation_map.shape
		if len(shape) == 4:
			activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
		elif len(shape) == 2:
			# try to make it square as much as possible. we can skip some activations.
			activations = activation_map[0]
			num_activations = len(activations)
			if num_activations > 1024:  # too hard to display it on the screen.
				square_param = int(np.floor(np.sqrt(num_activations)))
				activations = activations[0: square_param * square_param]
				activations = np.reshape(activations, (square_param, square_param))
			else:
				activations = np.expand_dims(activations, axis=0)
		else:
			raise Exception('len(shape) = 3 has not been implemented.')
		plt.imshow(activations, interpolation='None', cmap='gray')
		plt.show()

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
model.summary()
if (load):
	print("Loading model...")
	os.chdir(os.path.dirname(__file__))
	model.load_weights('mnist_28x28_percent98.96.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#print(K.learning_phase())
#K.set_learning_phase(0) #0 = test, 1 = train
#l1_activation = get_activations(model, x_test[0].reshape(1, 28, 28, 1), print_shape_only=True, layer_name='conv2d_1')
#display_activations(l1_activation)

'''output_fn = K.function([model.input], [model.layers[0].output])
output_image = output_fn([[x_test[0].reshape(28, 28, 1)]])

plt.imshow(output_image[0][0][:,:,2], interpolation='None', cmap='gray')
plt.show()'''

'''funcs = K.function([model.input] + [K.learning_phase()], [model.layers[0].output])
list_inputs = [x_test[0].reshape(1, 28, 28, 1), 0.]
layer_outputs = [funcs(list_inputs)[0]]
display_activations(layer_outputs)'''

from keras.models import Model

image = x_test[10,:,:,:].reshape(1, 28, 28, 1)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)
intermediate_output = intermediate_layer_model.predict(image)

plt.imshow(intermediate_output[0][:,:,1], interpolation='None', cmap='gray')
plt.show()

# https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
