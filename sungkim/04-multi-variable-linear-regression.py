import tensorflow as tf

x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data = [1, 2, 3, 4, 5]

W1 = tf.Variable(tf.random_uniform([1], -1, 1))
W2 = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))

y = W1 * x1_data + W2 * x2_data + b

cost = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 100 == 0:
        print step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)
