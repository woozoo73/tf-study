import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

images = np.reshape(mnist.train.images, [-1, 28, 28]);

plt.imshow(images[0], cmap='gray')
plt.show()
