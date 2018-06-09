from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import numpy as np

model = Sequential([
	Dense(2, input_dim=2, activation='relu'),
	Dense(2, activation='relu'),
	Dense(1, activation='sigmoid')
])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_y = np.array([[0], [1], [1], [1]])

print('OR gate implemented by neural network')
model.fit(train_x, train_y, batch_size=1, epochs=1000, verbose=0)

print(model.predict_proba(train_x))

print('[0, 0] => ', model.predict(np.array([[0, 0]]))[0][0])
print('[0, 1] => ', model.predict(np.array([[0, 1]]))[0][0])
print('[1, 0] => ', model.predict(np.array([[1, 0]]))[0][0])
print('[1, 1] => ', model.predict(np.array([[1, 1]]))[0][0])
