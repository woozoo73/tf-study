import tensorflow as tf

x_data = [
          [1., 1., 1., 1., 1.],
          [1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]

y_data = [1, 2, 3, 4, 5]

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
