import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W0 = tf.Variable(tf.random_normal([784, 256]))
b0 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(x, W0) + b0)

W1 = tf.Variable(tf.random_normal([256, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 10]))
b2 = tf.Variable(tf.random_normal([10]))
y = tf.matmul(L2, W2) + b2

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

# ('Epoch: ', 0, 'cost: ', 54.91328)
# ('Epoch: ', 1, 'cost: ', 49.146503)
# ('Epoch: ', 2, 'cost: ', 24.106653)
# ('Epoch: ', 3, 'cost: ', 18.531012)
# ('Epoch: ', 4, 'cost: ', 9.2214012)
# ('Epoch: ', 5, 'cost: ', 2.8677533)
# ('Epoch: ', 6, 'cost: ', 5.7535963)
# ('Epoch: ', 7, 'cost: ', 0.18099663)
# ('Epoch: ', 8, 'cost: ', 9.7238388)
# ('Epoch: ', 9, 'cost: ', 2.5347779)
# ('Epoch: ', 10, 'cost: ', 0.92838377)
# ('Epoch: ', 11, 'cost: ', 0.11858161)
# ('Epoch: ', 12, 'cost: ', 0.92640382)
# ('Epoch: ', 13, 'cost: ', 2.0666883)
# ('Epoch: ', 14, 'cost: ', 0.0)
# ('Epoch: ', 15, 'cost: ', 0.00052169268)
# ('Epoch: ', 16, 'cost: ', 0.0)
# ('Epoch: ', 17, 'cost: ', 2.6007385)
# ('Epoch: ', 18, 'cost: ', 0.0)
# ('Epoch: ', 19, 'cost: ', 0.19136901)
# ('Epoch: ', 20, 'cost: ', 1.4654384e-05)
# ('Epoch: ', 21, 'cost: ', 2.503392e-08)
# ('Epoch: ', 22, 'cost: ', 0.82439208)
# ('Epoch: ', 23, 'cost: ', 0.0)
# ('Epoch: ', 24, 'cost: ', 0.0)
# ('Epoch: ', 25, 'cost: ', 0.0)
# ('Epoch: ', 26, 'cost: ', 0.0)
# ('Epoch: ', 27, 'cost: ', 0.0)
# ('Epoch: ', 28, 'cost: ', 0.0)
# ('Epoch: ', 29, 'cost: ', 1.1920929e-09)
# 0.9563
