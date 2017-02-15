import tensorflow as tf

x_data = tf.placeholder(tf.float32)
y_data = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform ([1]))
b = tf.Variable(tf.zeros([1]))

y = W * x_data + b
cost = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={x_data: [1, 2, 3], y_data: [3, 5, 8]})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={x_data: [1, 2, 3], y_data: [2, 4, 6]}), sess.run(W), sess.run(b)
