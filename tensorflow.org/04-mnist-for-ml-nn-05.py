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

W9 = tf.get_variable("W9", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([256]))
L10 = tf.nn.dropout(tf.nn.relu(tf.matmul(L9, W9) + b9), keep_prob)

W10 = tf.get_variable("W10", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([256]))
L11 = tf.nn.dropout(tf.nn.relu(tf.matmul(L10, W10) + b10), keep_prob)

W11 = tf.get_variable("W11", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b11 = tf.Variable(tf.random_normal([256]))
L12 = tf.nn.dropout(tf.nn.relu(tf.matmul(L11, W11) + b11), keep_prob)

W12 = tf.get_variable("W12", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b12 = tf.Variable(tf.random_normal([256]))
L13 = tf.nn.dropout(tf.nn.relu(tf.matmul(L12, W12) + b12), keep_prob)

W13 = tf.get_variable("W13", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b13 = tf.Variable(tf.random_normal([256]))
L14 = tf.nn.dropout(tf.nn.relu(tf.matmul(L13, W13) + b13), keep_prob)

W14 = tf.get_variable("W14", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b14 = tf.Variable(tf.random_normal([256]))
L15 = tf.nn.dropout(tf.nn.relu(tf.matmul(L14, W14) + b14), keep_prob)

W15 = tf.get_variable("W15", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b15 = tf.Variable(tf.random_normal([256]))
L16 = tf.nn.dropout(tf.nn.relu(tf.matmul(L15, W15) + b15), keep_prob)

W16 = tf.get_variable("W16", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b16 = tf.Variable(tf.random_normal([256]))
L17 = tf.nn.dropout(tf.nn.relu(tf.matmul(L16, W16) + b16), keep_prob)

W17 = tf.get_variable("W17", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b17 = tf.Variable(tf.random_normal([256]))
L18 = tf.nn.dropout(tf.nn.relu(tf.matmul(L17, W17) + b17), keep_prob)

W18 = tf.get_variable("W18", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b18 = tf.Variable(tf.random_normal([256]))
L19 = tf.nn.dropout(tf.nn.relu(tf.matmul(L18, W18) + b18), keep_prob)

W19 = tf.get_variable("W19", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
b19 = tf.Variable(tf.random_normal([10]))
y = tf.matmul(L19, W19) + b19

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
