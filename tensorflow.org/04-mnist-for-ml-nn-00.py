import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    for step in range(total_batch):
        batch = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch[0], y_: batch[1]})
    if epoch % display_step == 0:
        print("Epoch: ", epoch, "cost: ", sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1]}))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})

# ('Epoch: ', 0, 'cost: ', 2.2482095)
# ('Epoch: ', 1, 'cost: ', 1.8534437)
# ('Epoch: ', 2, 'cost: ', 1.1261076)
# ('Epoch: ', 3, 'cost: ', 1.2995948)
# ('Epoch: ', 4, 'cost: ', 1.0680684)
# ('Epoch: ', 5, 'cost: ', 0.5946092)
# ('Epoch: ', 6, 'cost: ', 0.83005452)
# ('Epoch: ', 7, 'cost: ', 0.78425086)
# ('Epoch: ', 8, 'cost: ', 0.54540044)
# ('Epoch: ', 9, 'cost: ', 0.66613525)
# ('Epoch: ', 10, 'cost: ', 0.63556635)
# ('Epoch: ', 11, 'cost: ', 0.52399945)
# ('Epoch: ', 12, 'cost: ', 0.57550353)
# ('Epoch: ', 13, 'cost: ', 0.44633752)
# ('Epoch: ', 14, 'cost: ', 0.36297184)
# ('Epoch: ', 15, 'cost: ', 0.273893)
# ('Epoch: ', 16, 'cost: ', 0.34539634)
# ('Epoch: ', 17, 'cost: ', 0.39209116)
# ('Epoch: ', 18, 'cost: ', 0.21479355)
# ('Epoch: ', 19, 'cost: ', 0.42027366)
# ('Epoch: ', 20, 'cost: ', 0.28836185)
# ('Epoch: ', 21, 'cost: ', 0.31906417)
# ('Epoch: ', 22, 'cost: ', 0.59429842)
# ('Epoch: ', 23, 'cost: ', 0.2840139)
# ('Epoch: ', 24, 'cost: ', 0.36921585)
# ('Epoch: ', 25, 'cost: ', 0.20718846)
# ('Epoch: ', 26, 'cost: ', 0.36216295)
# ('Epoch: ', 27, 'cost: ', 0.31509587)
# ('Epoch: ', 28, 'cost: ', 0.34480828)
# ('Epoch: ', 29, 'cost: ', 0.20845814)
# 0.912
