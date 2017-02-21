import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W0 = tf.get_variable("W0", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b0 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(x, W0) + b0)

W1 = tf.get_variable("W1", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W1) + b1)

W2 = tf.get_variable("W2", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
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

# ('Epoch: ', 0, 'cost: ', 0.20675085)
# ('Epoch: ', 1, 'cost: ', 0.15532652)
# ('Epoch: ', 2, 'cost: ', 0.0766882)
# ('Epoch: ', 3, 'cost: ', 0.057996236)
# ('Epoch: ', 4, 'cost: ', 0.013406162)
# ('Epoch: ', 5, 'cost: ', 0.025763506)
# ('Epoch: ', 6, 'cost: ', 0.010298142)
# ('Epoch: ', 7, 'cost: ', 0.02240704)
# ('Epoch: ', 8, 'cost: ', 0.0042000618)
# ('Epoch: ', 9, 'cost: ', 0.0076550315)
# ('Epoch: ', 10, 'cost: ', 0.0070961071)
# ('Epoch: ', 11, 'cost: ', 0.0022533273)
# ('Epoch: ', 12, 'cost: ', 0.0042445627)
# ('Epoch: ', 13, 'cost: ', 0.0014748659)
# ('Epoch: ', 14, 'cost: ', 0.0016376248)
# ('Epoch: ', 15, 'cost: ', 0.00048010141)
# ('Epoch: ', 16, 'cost: ', 0.00013010648)
# ('Epoch: ', 17, 'cost: ', 0.0013604148)
# ('Epoch: ', 18, 'cost: ', 0.014602308)
# ('Epoch: ', 19, 'cost: ', 0.00071514311)
# ('Epoch: ', 20, 'cost: ', 0.0017645411)
# ('Epoch: ', 21, 'cost: ', 0.00052271417)
# ('Epoch: ', 22, 'cost: ', 0.0012281191)
# ('Epoch: ', 23, 'cost: ', 0.00044103523)
# ('Epoch: ', 24, 'cost: ', 9.1779082e-05)
# ('Epoch: ', 25, 'cost: ', 0.017311651)
# ('Epoch: ', 26, 'cost: ', 0.0006175281)
# ('Epoch: ', 27, 'cost: ', 0.00032574334)
# ('Epoch: ', 28, 'cost: ', 0.001531867)
# ('Epoch: ', 29, 'cost: ', 0.0015182586)
# 0.9785
