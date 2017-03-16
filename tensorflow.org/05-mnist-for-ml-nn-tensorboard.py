import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with tf.name_scope("L1") as scope:
    W0 = tf.get_variable("W0", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
    b0 = tf.Variable(tf.random_normal([512]), name='b0')
    tf.summary.histogram('histogram', W0)
    tf.summary.histogram('histogram', b0)
    L1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W0) + b0), keep_prob)

with tf.name_scope("L2") as scope:
    W1 = tf.get_variable("W1", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([256]), name='b1')
    tf.summary.histogram('histogram', W1)
    tf.summary.histogram('histogram', b1)
    L2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L1, W1) + b1), keep_prob)

with tf.name_scope("L3") as scope:
    W2 = tf.get_variable("W2", shape=[256, 1024], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([1024]), name='b2')
    tf.summary.histogram('histogram', W2)
    tf.summary.histogram('histogram', b2)
    L3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L2, W2) + b2), keep_prob)

with tf.name_scope("L4") as scope:
    W3 = tf.get_variable("W3", shape=[1024, 256], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([256]), name='b3')
    tf.summary.histogram('histogram', W3)
    tf.summary.histogram('histogram', b3)
    L4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L3, W3) + b3), keep_prob)

with tf.name_scope("L5") as scope:
    W4 = tf.get_variable("W4", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([256]), name='b4')
    tf.summary.histogram('histogram', W4)
    tf.summary.histogram('histogram', b4)
    L5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L4, W4) + b4), keep_prob)

with tf.name_scope("L6") as scope:
    W5 = tf.get_variable("W5", shape=[256, 1024], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([1024]), name='b5')
    tf.summary.histogram('histogram', W5)
    tf.summary.histogram('histogram', b5)
    L6 = tf.nn.dropout(tf.nn.relu(tf.matmul(L5, W5) + b5), keep_prob)

with tf.name_scope("L7") as scope:
    W6 = tf.get_variable("W6", shape=[1024, 2048], initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([2048]), name='b6')
    tf.summary.histogram('histogram', W6)
    tf.summary.histogram('histogram', b6)
    L7 = tf.nn.dropout(tf.nn.relu(tf.matmul(L6, W6) + b6), keep_prob)

with tf.name_scope("L8") as scope:
    W7 = tf.get_variable("W7", shape=[2048, 512], initializer=tf.contrib.layers.xavier_initializer())
    b7 = tf.Variable(tf.random_normal([512]), name='b7')
    tf.summary.histogram('histogram', W7)
    tf.summary.histogram('histogram', b7)
    L8 = tf.nn.dropout(tf.nn.relu(tf.matmul(L7, W7) + b7), keep_prob)

with tf.name_scope("L9") as scope:
    W8 = tf.get_variable("W8", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
    b8 = tf.Variable(tf.random_normal([256]), name='b8')
    tf.summary.histogram('histogram', W8)
    tf.summary.histogram('histogram', b8)
    L9 = tf.nn.dropout(tf.nn.relu(tf.matmul(L8, W8) + b8), keep_prob)

with tf.name_scope("L10") as scope:
    W9 = tf.get_variable("W9", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
    b9 = tf.Variable(tf.random_normal([10]), name='b9')
    tf.summary.histogram('histogram', W9)
    tf.summary.histogram('histogram', b9)
    y = tf.matmul(L9, W9) + b9

with tf.name_scope('train'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter("/tmp/mnist_logs/train", sess.graph)

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    for step in range(total_batch):
        print(epoch * step) + step
        batch = mnist.train.next_batch(batch_size)
        summary, _ = sess.run([merged, train], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7})
        train_writer.add_summary(summary, ((epoch + 1) * step) + step)
    if epoch % display_step == 0:
        print("Epoch: ", epoch, "cost: ", sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7}))

print sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

# ('Epoch: ', 0, 'cost: ', 0.69246328)
# ('Epoch: ', 1, 'cost: ', 0.1968473)
# ('Epoch: ', 2, 'cost: ', 0.23233816)
# ('Epoch: ', 3, 'cost: ', 0.21775122)
# ('Epoch: ', 4, 'cost: ', 0.14545858)
# ('Epoch: ', 5, 'cost: ', 0.14018366)
# ('Epoch: ', 6, 'cost: ', 0.21357653)
# ('Epoch: ', 7, 'cost: ', 0.12070619)
# ('Epoch: ', 8, 'cost: ', 0.080596827)
# ('Epoch: ', 9, 'cost: ', 0.12428339)
# ('Epoch: ', 10, 'cost: ', 0.075657628)
# ('Epoch: ', 11, 'cost: ', 0.1107536)
# ('Epoch: ', 12, 'cost: ', 0.054341588)
# ('Epoch: ', 13, 'cost: ', 0.048080426)
# ('Epoch: ', 14, 'cost: ', 0.19175965)
# ('Epoch: ', 15, 'cost: ', 0.17060618)
# ('Epoch: ', 16, 'cost: ', 0.07814423)
# ('Epoch: ', 17, 'cost: ', 0.090735875)
# ('Epoch: ', 18, 'cost: ', 0.024466878)
# ('Epoch: ', 19, 'cost: ', 0.017590402)
# ('Epoch: ', 20, 'cost: ', 0.14661108)
# ('Epoch: ', 21, 'cost: ', 0.053768039)
# ('Epoch: ', 22, 'cost: ', 0.028723516)
# ('Epoch: ', 23, 'cost: ', 0.058275748)
# ('Epoch: ', 24, 'cost: ', 0.023513269)
# ('Epoch: ', 25, 'cost: ', 0.15279666)
# ('Epoch: ', 26, 'cost: ', 0.23020233)
# ('Epoch: ', 27, 'cost: ', 0.14143369)
# ('Epoch: ', 28, 'cost: ', 0.03511611)
# ('Epoch: ', 29, 'cost: ', 0.066177599)
# 0.9805
