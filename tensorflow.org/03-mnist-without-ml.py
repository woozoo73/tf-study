import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import mnist_show as ms

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

images_group = [[], [], [], [], [], [], [], [], [], []]
for i, label in enumerate(mnist.train.labels):
    index = -1
    for j, v in enumerate(label):
        if v == 1:
            index = j
            break
    images_group[index].append(mnist.train.images[i])

images_sum = [[], [], [], [], [], [], [], [], [], []]
for i, images in enumerate(images_group):
    images_sum[i] = np.sum(images, 0)

images_sum_sum = [[], [], [], [], [], [], [], [], [], []]
for i, images in enumerate(images_sum):
    images_sum_sum[i] = np.sum(images, 0)

images_count = [0] * 10
for i, images in enumerate(images_group):
    images_count[i] = len(images)

images_w = [[], [], [], [], [], [], [], [], [], []]
for i, images in enumerate(images_sum):
    images_w[i] = images / (images_sum_sum[i] * images_count[i])

# for i, images in enumerate(images_w):
#     print np.sum(images) * images_count[i]

ms.show_mnist_images(np.reshape(images_w, [-1, 28, 28]), row_size=2, col_size=5)

# transpose: [10, 784] --> [784, 10]
W = np.array(images_w).T

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
y = tf.matmul(x, W)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

print sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})

predictions = sess.run(correct_prediction, {x: mnist.test.images, y_: mnist.test.labels})
p_labels = sess.run(tf.argmax(y, 1), {x: mnist.test.images})
labels = sess.run(tf.argmax(mnist.test.labels, 1))

print p_labels

correct_labels = []
correct_images = []
incorrect_labels = []
incorrect_images = []

for i in range(len(predictions)):
    label = str(labels[i]) + '-->' + str(p_labels[i])
    image = np.reshape(mnist.test.images[i], [-1, 28, 28])
    if predictions[i]:
        correct_labels.append(label)
        correct_images.append(image)
    else:
        incorrect_labels.append(label)
        incorrect_images.append(image)

ms.show_mnist_images(correct_images, correct_labels, title='Correct predictions #' + str(len(correct_images)))
ms.show_mnist_images(incorrect_images, incorrect_labels, title='Incorrect predictions #' + str(len(incorrect_images)))

