from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

import numpy as np

model = Sequential([
	Dense(2, input_dim=2, activation='relu', name='input'),
	#Dense(2, activation='relu', name='layer_1'),
	Dense(1, activation='sigmoid', name='output')
])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)
#model.summary()

'''model.set_weights([
	# Input
	[
		[-0.6, 3.6],
		[-2., 2.8]
	],
	[0., 0.],

	# Output
	[
		[ 2.],
		[ 4.5]
	],
	[-6.5]
])'''

train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_y = np.array([[0], [1], [1], [1]])

print('OR gate implemented by neural network')
tbCallBack = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_images=True)
model.fit(train_x, train_y, batch_size=1, epochs=1000, verbose=0, callbacks=[tbCallBack])
# tensorboard --logdir=log

print(model.predict_proba(train_x))

print('[0, 0] => ', model.predict(np.array([[0, 0]]))[0][0])
print('[0, 1] => ', model.predict(np.array([[0, 1]]))[0][0])
print('[1, 0] => ', model.predict(np.array([[1, 0]]))[0][0])
print('[1, 1] => ', model.predict(np.array([[1, 1]]))[0][0])

# Layer weight
'''for layer in model.layers:
	weights = layer.get_weights()
	print(layer.name + ':')
	print(weights)'''

print(model.get_weights())