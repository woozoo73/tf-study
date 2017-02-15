import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

x_data = xy[0:-1]
y_data = xy[-1]

# print xy
# print x_data
# print y_data

W = tf.Variable(tf.random_uniform([1, 3], -1, 1))

y = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(y_data - y))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 100 == 0:
        print step, sess.run(cost), sess.run(W)
