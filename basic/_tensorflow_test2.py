import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.Variable(tf.random_normal((D, H)))
w2 = tf.Variable(tf.random_normal((H, D)))

h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
loss = tf.losses.mean_squared_error(y_pred, y)

optimizer = tf.train.GradientDescentOptimizer(1e-3)
#optimizer = tf.train.AdamOptimizer()
updates = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	values = {x: np.random.randn(N, D), y: np.random.randn(N, D)}

	losses = []
	for t in range(50):
		loss_val, _ = sess.run([loss, updates], feed_dict=values)
		losses.append(loss_val)

	plt.plot(losses)
	plt.ylabel('some numbers')
	plt.show()
