from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop

import numpy as np

model = Sequential()
model.add(Conv1D(2, 2, input_shape=(2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

#model.summary

optimizer = RMSprop(lr=1e-5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# prediction
test = np.array([0, 1])
predictions = model.predict(test, verbose=1)

'''import numpy
from keras.models import Sequential
from keras.layers.convolutional import Conv1D

numpy.random.seed(7)

model = Sequential()
model.add(Conv1D(2, 2, activation='relu', input_shape=X.shape))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=5)
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))'''
