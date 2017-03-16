import numpy as np
import matplotlib.pyplot as plt


def show_mnist_images(images, labels=None, row=10, col=10, w=28, h=28, title='Numbers', cmap='gray_r'):
    images = np.reshape(images, [-1, w, h])

    fig, axes = plt.subplots(row, col)
    fig.suptitle(title)

    for i in range(row):
        for j in range(col):
            ax = None
            if row > 1 and col > 1:
                ax = axes[i, j]
            elif row > 1:
                ax = axes[i]
            elif col > 1:
                ax = axes[j]
            x = i * col + j
            if x < len(images):
                image = images[x]
                ax.imshow(image, cmap=cmap)
                if labels is not None:
                    ax.text(0, 0, labels[x])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
            else:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

    plt.show()
