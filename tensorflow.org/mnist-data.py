import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

sess = tf.Session()

sum_of_train_labels = tf.reduce_sum(mnist.train.labels, 0)
sum_of_validation_labels = tf.reduce_sum(mnist.validation.labels, 0)
sum_of_test_labels = tf.reduce_sum(mnist.test.labels, 0)

print(sess.run(sum_of_train_labels))
print(sess.run(sum_of_validation_labels))
print(sess.run(sum_of_test_labels))

images = np.reshape(mnist.train.images, [-1, 28, 28]);
nega_image = np.reshape([1.] * 28 * 28, [28, 28])
nega_images = nega_image - images
labels = mnist.train.labels

row_size = 10
col_size = 10

fig, axes = plt.subplots(row_size, col_size)

for i in range(row_size):
    for j in range(col_size):
        index = i * row_size + j
        print index
        ax = axes[i, j]
        ax.imshow(nega_images[index], cmap='gray')
        ax.axis('off')
        ax.text(0, 0, labels[index])

plt.subplots_adjust(wspace=0, hspace=0, left=None, right=None, bottom=None, top=None)
plt.show()
