import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


def show_mnist_images(images, labels=None, row_size=10, col_size=10, title='Numbers'):
    images = np.reshape(images, [-1, 28, 28]);
    mask_image = np.reshape([1.] * 28 * 28, [28, 28])
    nega_images = mask_image - images

    fig, axes = plt.subplots(row_size, col_size)
    fig.suptitle(title)

    for i in range(row_size):
        for j in range(col_size):
            ax = None
            if row_size > 1 and col_size > 1:
                ax = axes[i, j]
            elif row_size > 1:
                ax = axes[i]
            elif col_size > 1:
                ax = axes[j]
            x = i * col_size + j
            if x < len(images):
                ax.imshow(nega_images[x], cmap='gray')
            if labels is not None:
                ax.text(0, 0, labels[x])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    plt.tick_params()
    plt.show()
