import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
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

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

ms.show_mnist_images(np.array(sess.run(W)).T, title='W', row=2, col=5)

print sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})

predictions = sess.run(correct_prediction, {x: mnist.test.images, y_: mnist.test.labels})
p_labels = sess.run(tf.argmax(y, 1), {x: mnist.test.images})
labels = sess.run(tf.argmax(mnist.test.labels, 1))

print p_labels

correct_labels, correct_images, incorrect_labels, incorrect_images = [], [], [], []
for i in range(len(predictions)):
    label = str(labels[i]) + '-->' + str(p_labels[i])
    image = mnist.test.images[i]
    if predictions[i]:
        correct_labels.append(label)
        correct_images.append(image)
    else:
        incorrect_labels.append(label)
        incorrect_images.append(image)

ms.show_mnist_images(correct_images, correct_labels, title='Correct predictions #' + str(len(correct_images)))
ms.show_mnist_images(incorrect_images, incorrect_labels, title='Incorrect predictions #' + str(len(incorrect_images)))
