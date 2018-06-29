import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

class cnn_faces:
	def __init__(self, train=True):
		self.num_classes = 40
		self.weight_decay = 0.0005
		self.x_shape = [64, 64, 1]

		self.model = self.build_model()
		if train:
			self.model = self.train(self.model)
		else:
			self.model.load_weights('./faces_64x64.h5')

	def build_model(self):
		# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
		model = Sequential()
		weight_decay = self.weight_decay

		model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.3))

		model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))

		model.add(Flatten())
		model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(Dropout(0.5))
		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))
		#model.summary()
		return model

	def normalize(self, X_train, X_test):
		#this function normalize inputs for zero mean and unit variance
		# it is used when training a model.
		# Input: training set and test set
		# Output: normalized training set and test set according to the trianing set statistics.
		mean = np.mean(X_train, axis=(0, 1, 2, 3))
		std = np.std(X_train, axis=(0, 1, 2, 3))
		X_train = (X_train-mean)/(std+1e-7)
		X_test = (X_test-mean)/(std+1e-7)
		return X_train, X_test

	def normalize_production(self, x):
		#this function is used to normalize instances in production according to saved training set statistics
		# Input: X - a training set
		# Output X - a normalized training set according to normalization constants.

		#these values produced during first training and are general for the standard cifar10 training set normalization
		mean = 128
		std = 255
		return (x-mean)/(std+1e-7)

	def predict(self ,x, normalize=True, batch_size=50):
		if normalize:
			x = self.normalize_production(x)
		return self.model.predict(x, batch_size)

	'''def show_weights(self, layer, size):
			weights = self.model.layers[layer].get_weights()[0]
			print(self.model.layers[layer].name)
			#plt.title('Weights of the 1st convolutional layer')
			for i in range(size*size):
					plt.subplot(size, size, i+1)
					plt.imshow(weights[:, :, 0, i], cmap='gray', interpolation='none');
					plt.axis('off');

			plt.show()

	def show_activation_map(self, layer, size, image):
		#Create new sequential model, same as before but just keep the convolutional layer.
		model_new = Sequential()
		model_new.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=self.x_shape))
		model_new.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
		model_new.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

		print(self.model.layers[layer].name)
		model_new.layers[0].set_weights(self.model.layers[0].get_weights())
		model_new.layers[1].set_weights(self.model.layers[layer].get_weights())

		pred = model_new.predict(image.reshape(1, self.x_shape[0], self.x_shape[1], self.x_shape[2]))

		#For all the filters, plot the output of the input
		plt.figure(figsize=(size, size))
		filts = pred[0]
		for i in range(size*size):
			filter_digit = filts[:,:,i]
			plt.subplot(size, size, i+1)
			plt.imshow(filter_digit, cmap='jet'); plt.axis('off');

		plt.show()'''

	def train(self, model):
		#training parameters
		batch_size = 128
		maxepoches = 250
		learning_rate = 0.1
		lr_decay = 1e-6
		lr_drop = 20

		# The data, shuffled and split between train and test sets:
		oliv = fetch_olivetti_faces()
		x_train, x_test, y_train, y_test = train_test_split(oliv.images, oliv.target, test_size=0.2, random_state=42)

		x_train = x_train.reshape(x_train.shape[0], self.x_shape[0], self.x_shape[1], self.x_shape[2]).astype('float32')
		x_test = x_test.reshape(x_test.shape[0], self.x_shape[0], self.x_shape[1], self.x_shape[2]).astype('float32')
		x_train, x_test = self.normalize(x_train, x_test)

		y_train = keras.utils.to_categorical(y_train, self.num_classes)
		y_test = keras.utils.to_categorical(y_test, self.num_classes)

                # callbacks
		def lr_scheduler(epoch):
			return learning_rate * (0.5 ** (epoch // lr_drop))
		reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
		tbCallBack = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_images=True)
		# tensorboard --logdir=log
		filepath = 'faces_{epoch:02d}-{loss:.4f}.h5'
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=10)

		#data augmentation
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False)  # randomly flip images
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)

		#optimization details
		sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

		# training process in a for loop with learning rate drop every 25 epoches.
		historytemp = model.fit_generator(datagen.flow(x_train, y_train,
							batch_size=batch_size),
							steps_per_epoch=x_train.shape[0] // batch_size,
							epochs=maxepoches,
							validation_data=(x_test, y_test), callbacks=[reduce_lr, tbCallBack, checkpoint], verbose=1)
		# tensorboard --logdir=log
		model.save_weights('faces_64x64.h5')
		return model
	
'''def show_image(img_data):
	plt.figure(figsize=(10,5))
	plt.imshow(img_data, cmap='gray')
	plt.show()

#Showing the faces
fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(25):
	ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
	ax.imshow(oliv.images[i+100], cmap='gray', interpolation='nearest')
	ax.set_title(oliv.target[i+100])

plt.show()'''

if __name__ == '__main__':
	'''oliv = fetch_olivetti_faces()
	x_train, x_test, y_train, y_test = train_test_split(oliv.images, oliv.target, test_size=0.2, random_state=42)
	x_test = x_test.astype('float32')
	y_test = keras.utils.to_categorical(y_test, 40)'''

	model = cnn_faces(train=True)

	'''predicted_x = model.predict(x_test)
	residuals = np.argmax(predicted_x, 1)!=np.argmax(y_test, 1)

	loss = sum(residuals)/len(residuals)
	print("the validation 0/1 loss is: ", loss)'''

	'''#model.show_weights(layer=28, size=16)
	#i = np.random.randint(0, len(x_test))
	i = 10 # aeroplane 
	print('Random image index: ', i)
	show_image(x_test[i])
	#model.show_activation_map(layer=0, size=8, image=x_test[i])
	model.show_activation_map(layer=8, size=12, image=x_test[i])'''
