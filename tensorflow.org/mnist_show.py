import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


def show(images, c_labels, p_labels, indexes, row_size=10, col_size=10, title='Numbers'):
    images = np.reshape(images, [-1, 28, 28]);
    nega_image = np.reshape([1.] * 28 * 28, [28, 28])
    nega_images = nega_image - images

    fig, axes = plt.subplots(row_size, col_size)
    fig.suptitle(title)

    for i in range(row_size):
        for j in range(col_size):
            x = i * row_size + j
            if x >= len(indexes):
                break
            ax = axes[i, j]
            index = indexes[x]
            ax.imshow(nega_images[index], cmap='gray')
            ax.text(0, 0, '' + str(c_labels[index]) + ':' + str(p_labels[index]))

    for i in range(row_size):
        for j in range(col_size):
            ax = axes[i, j]
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    plt.tick_params()
    plt.show()
