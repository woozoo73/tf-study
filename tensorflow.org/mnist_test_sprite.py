import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

images = mnist.test.images[:1000]
sprite = images.reshape(-1, 28)
plt.imsave('mnist_test_sprite.png', -sprite, cmap=cm.gray)
