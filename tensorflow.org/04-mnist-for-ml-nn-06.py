import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W0 = tf.get_variable("W0", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b0 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W0) + b0), keep_prob)

W1 = tf.get_variable("W1", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L1, W1) + b1), keep_prob)

W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L2, W2) + b2), keep_prob)

W3 = tf.get_variable("W3", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L3, W3) + b3), keep_prob)

W4 = tf.get_variable("W4", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L4, W4) + b4), keep_prob)

W5 = tf.get_variable("W5", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([256]))
L6 = tf.nn.dropout(tf.nn.relu(tf.matmul(L5, W5) + b5), keep_prob)

W6 = tf.get_variable("W6", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([256]))
L7 = tf.nn.dropout(tf.nn.relu(tf.matmul(L6, W6) + b6), keep_prob)

W7 = tf.get_variable("W7", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([256]))
L8 = tf.nn.dropout(tf.nn.relu(tf.matmul(L7, W7) + b7), keep_prob)

W8 = tf.get_variable("W8", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([256]))
L9 = tf.nn.dropout(tf.nn.relu(tf.matmul(L8, W8) + b8), keep_prob)

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
        sess.run(train, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7})
    if epoch % display_step == 0:
        print("Epoch: ", epoch, "cost: ", sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7}))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

# ('Epoch: ', 0, 'cost: ', 0.63543743)
# ('Epoch: ', 1, 'cost: ', 0.36307281)
# ('Epoch: ', 2, 'cost: ', 0.12369938)
# ('Epoch: ', 3, 'cost: ', 0.22233254)
# ('Epoch: ', 4, 'cost: ', 0.25256661)
# ('Epoch: ', 5, 'cost: ', 0.14657253)
# ('Epoch: ', 6, 'cost: ', 0.070000984)
# ('Epoch: ', 7, 'cost: ', 0.096610352)
# ('Epoch: ', 8, 'cost: ', 0.19995615)
# ('Epoch: ', 9, 'cost: ', 0.096062802)
# ('Epoch: ', 10, 'cost: ', 0.075120986)
# ('Epoch: ', 11, 'cost: ', 0.10478239)
# ('Epoch: ', 12, 'cost: ', 0.020078648)
# ('Epoch: ', 13, 'cost: ', 0.091979638)
# ('Epoch: ', 14, 'cost: ', 0.042258523)
# ('Epoch: ', 15, 'cost: ', 0.039708279)
# ('Epoch: ', 16, 'cost: ', 0.13789497)
# ('Epoch: ', 17, 'cost: ', 0.072956562)
# ('Epoch: ', 18, 'cost: ', 0.19079283)
# ('Epoch: ', 19, 'cost: ', 0.13327721)
# ('Epoch: ', 20, 'cost: ', 0.038663264)
# ('Epoch: ', 21, 'cost: ', 0.088565752)
# ('Epoch: ', 22, 'cost: ', 0.044037972)
# ('Epoch: ', 23, 'cost: ', 0.09921211)
# ('Epoch: ', 24, 'cost: ', 0.096481301)
# ('Epoch: ', 25, 'cost: ', 0.12397099)
# ('Epoch: ', 26, 'cost: ', 0.042631526)
# ('Epoch: ', 27, 'cost: ', 0.026588175)
# ('Epoch: ', 28, 'cost: ', 0.11573672)
# ('Epoch: ', 29, 'cost: ', 0.038175315)
# 0.9799
