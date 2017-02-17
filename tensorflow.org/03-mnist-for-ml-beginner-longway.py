import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
mnist = input_data.read_data_sets("MNIST_data")
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b
#   (?, 784) * (784 * 10) + (10, ?)
# = (? * 10)              + (10, ?) # matmul
# = (? * 10)              + (?, 10) # broadcasting

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

sess.run(tf.global_variables_initializer())

for step in range(1000):
    batch = mnist.train.next_batch(200)
    xm = batch[0]
    ym = []
    for yu in batch[1]:
        ymu = [0] * 10
        ymu[yu] = 1
        ym.append(ymu)
    data = {x: xm, y_: ym}
    train.run(feed_dict=data)
    if step % 100 == 0:
        sess.run(y, feed_dict=data)

xm = mnist.test.images
data = {x: xm}

predicts = sess.run(y, {x:xm})
print("predicts.shape", predicts.shape)
print("predicts", predicts)

reals = mnist.test.labels

success = 0
failure = 0
for i, predict in enumerate(predicts):
    maxval = None
    max = -1
    for j, v in enumerate(predict):
        if (max == -1) or (maxval < v):
            maxval = v
            max = j
    if max == reals[i]:
        success = success + 1
    else:
        failure = failure + 1
    print(reals[i], max)

print(success, failure, (1.0 * success) / (success + failure))
