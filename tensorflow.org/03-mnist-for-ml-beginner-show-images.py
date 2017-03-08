import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import mnist_show as ms

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
optimizer = tf.train.GradientDescentOptimizer(0.5)
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

predictions = sess.run(correct_prediction, {x: mnist.test.images, y_: mnist.test.labels})
p_labels = sess.run(tf.argmax(y, 1), {x: mnist.test.images})

print p_labels

correct_indexes = []
incorrect_indexes = []
for p_index in range(len(predictions)):
    if predictions[p_index]:
        correct_indexes.append(p_index)
    else:
        incorrect_indexes.append(p_index)

print("incorrect_indexes : %s" % incorrect_indexes)

images = np.reshape(mnist.test.images, [-1, 28, 28]);
nega_image = np.reshape([1.] * 28 * 28, [28, 28])
nega_images = nega_image - images
labels = sess.run(tf.argmax(mnist.test.labels, 1))

ms.show(images, labels, p_labels, correct_indexes, title='Correct predictions #' + str(len(correct_indexes)))
ms.show(images, labels, p_labels, incorrect_indexes[0:17], title='Incorrect predictions #' + str(len(incorrect_indexes)))
