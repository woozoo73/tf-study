import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def get_test_image(i):
    images = []
    image = mnist.test.images[i]
    images.append(image)
    return images


def get_test_label(i):
    labels = []
    label = mnist.test.labels[i]
    labels.append(label)
    return labels

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# print mnist.test.images[0]
# print mnist.test.labels[0]

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print(y)
print(cross_entropy)
print(W)
print(b)
print(x)
print(sess.run(W))
print('W[000]: %s' % sess.run(W[0]))
print('W[392]: %s' % sess.run(W[392]))
print(sess.run(b))

"""
for step in range(len(mnist.test.images)):
    test_images = get_test_image(step)
    test_labels = get_test_label(step)
    test_label = mnist.test.labels[step]

    # print(sess.run(y, feed_dict={x: test_images}))
    # print(sess.run(tf.nn.softmax(y), feed_dict={x: test_images}))
    # print(sess.run(tf.argmax(y, 1), feed_dict={x: test_images}))
    # print(sess.run(tf.argmin(y, 1), feed_dict={x: test_images}))
    # print(sess.run(tf.argmin(y, 0), feed_dict={x: test_images}))
    # print(sess.run(tf.argmax(y_, 1), feed_dict={y_: test_labels}))

    sess.run(tf.nn.softmax(y), feed_dict={x: test_images})
"""

results = sess.run(y, feed_dict={x: mnist.test.images})


def max_index(value):
    i = -1
    cv = None
    for index in xrange(len(value)):
        if (cv == None or cv < value[index]):
            cv = value[index]
            i = index
    return i

answers_x = []
answers_y = []

for row in xrange(len(results)):
    index = max_index(results[row])
    value = results[row][index]
    correct = max_index(mnist.test.labels[row])
    answers_x.append(value)
    if (index == correct):
        answers_y.append(1)
    else:
        answers_y.append(0)

# print answers_x
# print answers_y

plt.hist(answers_x)
plt.show()
