# Original paper
#https://deeplearning4j.org/generative-adversarial-network
# Github source
#https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt

import sys, glob, os

import numpy as np

class GAN():
	def __init__(self):
		self.img_rows = 150 
		self.img_cols = 150
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		optimizer = Adam(0.0002, 0.5)

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		# Build and compile the generator
		self.generator = self.build_generator()
		self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

		# The generator takes noise as input and generated imgs
		z = Input(shape=(150,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The valid takes generated images as input and determines validity
		valid = self.discriminator(img)

		# The combined model  (stacked generator and discriminator) takes
		# noise as input => generates images => determines validity 
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_generator(self):
		noise_shape = (150,)

		model = Sequential()

		model.add(Dense(256, input_shape=noise_shape))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(self.img_shape), activation='tanh'))
		model.add(Reshape(self.img_shape))

		#model.summary()

		noise = Input(shape=noise_shape)
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self):
		model = Sequential()

		model.add(Flatten(input_shape=self.img_shape))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(256))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, activation='sigmoid'))
		#model.summary()

		img = Input(shape=self.img_shape)
		validity = model(img)

		return Model(img, validity)

	def load_images(self, path):
		data = []

		os.chdir(path)
		for file in glob.glob("*.jpg"):
			img_array = img_to_array(load_img(file, target_size=(self.img_rows, self.img_cols)))
			data.append(img_array)

		return np.array(data)

	def train(self, epochs, batch_size=128, save_interval=50):
		# Load the dataset
		#(X_train, _), (_, _) = mnist.load_data()

		print('Loading images...')
		os.chdir(os.path.dirname(__file__))
		X_train = self.load_images('gan-face\input')

		print('Loaded %d instances of training data' % X_train.shape[0])

		# Rescale -1 to 1
		X_train = (X_train.astype(np.float32) - 127.5) / 127.5

		half_batch = int(batch_size / 2)

		for epoch in range(epochs):
			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Select a random half batch of images
			idx = np.random.randint(0, X_train.shape[0], half_batch)
			imgs = X_train[idx]

			noise = np.random.normal(0, 1, (half_batch, 150))
			# Generate a half batch of new images
			gen_imgs = self.generator.predict(noise)

			# Train the discriminator
			d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------
			noise = np.random.normal(0, 1, (batch_size, 150))

			# The generator wants the discriminator to label the generated samples
			# as valid (ones)
			valid_y = np.array([1] * batch_size)

			# Train the generator
			g_loss = self.combined.train_on_batch(noise, valid_y)

			# Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			# If at save interval => save generated image samples
			if epoch % save_interval == 0:
				self.save_imgs(epoch)
				#self.show_grayscale_image(epoch)'''

	'''def show_grayscale_image(self, img_data):
		noise = np.random.normal(0, 1, (1, 100))
		gen_imgs = self.generator.predict(noise)

		plt.figure(figsize=(10,5))
		plt.imshow(np.reshape(gen_imgs[0, :,:,0], (self.img_rows, self.img_cols)).astype(np.uint8), cmap=plt.cm.gray)
		plt.show()'''

	def save_imgs(self, epoch):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, 150))
		gen_imgs = self.generator.predict(noise)

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:]) #, cmap='gray'
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig(os.path.dirname(__file__) + "\gan-face\images\mnist_%d.png" % epoch)
		plt.close()

if __name__ == '__main__':
	gan = GAN()
	gan.train(epochs=10000, batch_size=32, save_interval=100)
