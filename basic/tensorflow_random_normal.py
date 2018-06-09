import tensorflow as tf

gauss = tf.Variable(tf.random_normal((1000,1), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(gauss))
