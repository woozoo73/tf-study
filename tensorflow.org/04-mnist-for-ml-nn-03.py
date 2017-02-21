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

W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L3 = tf.nn.relu(tf.matmul(L2, W2) + b2)

W3 = tf.get_variable("W3", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.relu(tf.matmul(L3, W3) + b3)

W4 = tf.get_variable("W4", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L5 = tf.nn.relu(tf.matmul(L4, W4) + b4)

W5 = tf.get_variable("W5", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([256]))
L6 = tf.nn.relu(tf.matmul(L5, W5) + b5)

W6 = tf.get_variable("W6", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([256]))
L7 = tf.nn.relu(tf.matmul(L6, W6) + b6)

W7 = tf.get_variable("W7", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([256]))
L8 = tf.nn.relu(tf.matmul(L7, W7) + b7)

W8 = tf.get_variable("W8", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([256]))
L9 = tf.nn.relu(tf.matmul(L8, W8) + b8)

W9 = tf.get_variable("W9", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([10]))

y = tf.matmul(L9, W9) + b9

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
        print("Epoch: ", epoch, "cost: ", sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1]}, ))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})

# ('Epoch: ', 0, 'cost: ', 0.16505235)
# ('Epoch: ', 1, 'cost: ', 0.10212722)
# ('Epoch: ', 2, 'cost: ', 0.11817381)
# ('Epoch: ', 3, 'cost: ', 0.056148477)
# ('Epoch: ', 4, 'cost: ', 0.066372521)
# ('Epoch: ', 5, 'cost: ', 0.012768423)
# ('Epoch: ', 6, 'cost: ', 0.038609736)
# ('Epoch: ', 7, 'cost: ', 0.15476853)
# ('Epoch: ', 8, 'cost: ', 0.0054428563)
# ('Epoch: ', 9, 'cost: ', 0.061613753)
# ('Epoch: ', 10, 'cost: ', 0.013653398)
# ('Epoch: ', 11, 'cost: ', 0.0038398234)
# ('Epoch: ', 12, 'cost: ', 0.03780999)
# ('Epoch: ', 13, 'cost: ', 0.005806223)
# ('Epoch: ', 14, 'cost: ', 0.055255976)
# ('Epoch: ', 15, 'cost: ', 0.0066758776)
# ('Epoch: ', 16, 'cost: ', 0.016501438)
# ('Epoch: ', 17, 'cost: ', 0.045018405)
# ('Epoch: ', 18, 'cost: ', 0.0014824931)
# ('Epoch: ', 19, 'cost: ', 0.017630171)
# ('Epoch: ', 20, 'cost: ', 0.002194375)
# ('Epoch: ', 21, 'cost: ', 0.0042774039)
# ('Epoch: ', 22, 'cost: ', 0.0037616133)
# ('Epoch: ', 23, 'cost: ', 0.0023427634)
# ('Epoch: ', 24, 'cost: ', 0.060201444)
# ('Epoch: ', 25, 'cost: ', 0.0040147081)
# ('Epoch: ', 26, 'cost: ', 0.0001957248)
# ('Epoch: ', 27, 'cost: ', 0.044230863)
# ('Epoch: ', 28, 'cost: ', 0.0044386555)
# ('Epoch: ', 29, 'cost: ', 0.01001724)
# 0.9773
