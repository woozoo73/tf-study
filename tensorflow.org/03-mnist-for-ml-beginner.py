import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

print(W)
print(b)

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# pred = tf.nn.softmax(y)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(pred), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(0.5)
# optimizer = tf.train.AdamOptimizer(0.001)
# optimizer = tf.train.AdadeltaOptimizer(10)
train = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    batch = mnist.train.next_batch(100)
    sess.run(train, {x: batch[0], y_: batch[1]})
    if step % 100 == 0:
        sess.run(y, feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})
