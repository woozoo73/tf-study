import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
import os

PATH_TO_MNIST_DATA = "MNIST_data"
LOG_DIR = "/home/lab/PycharmProjects/tf-study/tensorflow.org/log"
IMAGE_NUM = 2

# Read in MNIST data by utility functions provided by TensorFlow
mnist = input_data.read_data_sets(PATH_TO_MNIST_DATA, one_hot=False)

# Extract target MNIST image data
# plot_array = mnist.test.images[:IMAGE_NUM]  # shape: (n_observations, n_features)

session = tf.InteractiveSession()

# Generate meta data
np.savetxt(os.path.join(LOG_DIR, 'metadata.tsv'), mnist.test.labels[:IMAGE_NUM], fmt='%d')

# Download sprite image
# https://www.tensorflow.org/images/mnist_10k_sprite.png, 100x100 thumbnails

# To visualise your embeddings, there are 3 things you need to do:
# 1) Setup a 2D tensor variable(s) that holds your embedding(s)

# embedding_var = tf.Variable(plot_array, name='embedding')
with tf.device("/cpu:0"):
    embedding = tf.Variable(tf.stack(mnist.test.images[:IMAGE_NUM], axis=0), trainable=False, name='embedding')

tf.global_variables_initializer().run()

# 2) Periodically save your embeddings in a LOG_DIR
# Here we just save the Tensor once, so we set global_step to a fixed number
saver = tf.train.Saver()

# 3) Associate metadata and sprite image with your embedding
# Use the same LOG_DIR where you stored your checkpoint.
writer = tf.summary.FileWriter(LOG_DIR, session.graph)

config = projector.ProjectorConfig()
# You can add multiple embeddings. Here we add only one.
embed = config.embeddings.add()
embed.tensor_name = 'embedding:0' # embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embed.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
# Link this tensor to its sprite image.
embed.sprite.image_path = os.path.join(LOG_DIR, 'mnist_10k_sprite.png')
embed.sprite.single_image_dim.extend([28, 28])
# Saves a configuration file that TensorBoard will read during startup.
projector.visualize_embeddings(writer, config)

saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), global_step=IMAGE_NUM)
