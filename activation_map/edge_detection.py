# https://sempwn.github.io/blog/2017/04/06/conv_net_intro
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

i = np.random.randint(x_train.shape[0])

c = x_train[i,:,:]
# https://en.wikipedia.org/wiki/Kernel_%28image_processing%29
k = [
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]
]

# Sample convolution
# c = signal.convolve2d([[1,0,1],[0,1,0],[1,0,1]], [[1,-1],[-1,1]], boundary='fill', mode='valid');
# [[2,0], [0,2]]

plt.figure();
plt.subplot(1, 2, 1);
plt.imshow(c,cmap='gray'); plt.axis('off');
plt.title('original image');

plt.subplot(1, 2, 2);
c_digit = signal.convolve2d(c, k, boundary='symm', mode='same');
plt.imshow(c_digit, cmap='gray');
plt.axis('off');
plt.title('edge-detection image');   

plt.show()