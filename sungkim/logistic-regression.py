import tensorflow as tf

x_data = [
    [1., 1., 1., 1., 1., 1.],
    [2., 3., 3., 5., 7., 2.],
    [1., 2., 4., 5., 5., 5.]]
y_data = [0., 0., 0., 1., 1., 1.]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

h = tf.matmul(W, X)
y = tf.div(1., 1. + tf.exp(-h))
# cost = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y)))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

print sess.run(y, feed_dict={X: [[1], [2], [2]]}) > 0.5
print sess.run(y, feed_dict={X: [[1], [5], [5]]}) > 0.5

print sess.run(y, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}) > 0.5
