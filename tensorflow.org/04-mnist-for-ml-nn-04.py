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

W9 = tf.get_variable("W9", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([256]))
L10 = tf.nn.relu(tf.matmul(L9, W9) + b9)

W10 = tf.get_variable("W10", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([256]))
L11 = tf.nn.relu(tf.matmul(L10, W10) + b10)

W11 = tf.get_variable("W11", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b11 = tf.Variable(tf.random_normal([256]))
L12 = tf.nn.relu(tf.matmul(L11, W11) + b11)

W12 = tf.get_variable("W12", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b12 = tf.Variable(tf.random_normal([256]))
L13 = tf.nn.relu(tf.matmul(L12, W12) + b12)

W13 = tf.get_variable("W13", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b13 = tf.Variable(tf.random_normal([256]))
L14 = tf.nn.relu(tf.matmul(L13, W13) + b13)

W14 = tf.get_variable("W14", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b14 = tf.Variable(tf.random_normal([256]))
L15 = tf.nn.relu(tf.matmul(L14, W14) + b14)

W15 = tf.get_variable("W15", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b15 = tf.Variable(tf.random_normal([256]))
L16 = tf.nn.relu(tf.matmul(L15, W15) + b15)

W16 = tf.get_variable("W16", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b16 = tf.Variable(tf.random_normal([256]))
L17 = tf.nn.relu(tf.matmul(L16, W16) + b16)

W17 = tf.get_variable("W17", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b17 = tf.Variable(tf.random_normal([256]))
L18 = tf.nn.relu(tf.matmul(L17, W17) + b17)

W18 = tf.get_variable("W18", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b18 = tf.Variable(tf.random_normal([256]))
L19 = tf.nn.relu(tf.matmul(L18, W18) + b18)

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
        sess.run(train, feed_dict={x: batch[0], y_: batch[1]})
    if epoch % display_step == 0:
        print("Epoch: ", epoch, "cost: ", sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1]}, ))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})

# ('Epoch: ', 0, 'cost: ', 0.56883538)
# ('Epoch: ', 1, 'cost: ', 0.37584758)
# ('Epoch: ', 2, 'cost: ', 0.4310106)
# ('Epoch: ', 3, 'cost: ', 0.17758633)
# ('Epoch: ', 4, 'cost: ', 0.29892841)
# ('Epoch: ', 5, 'cost: ', 0.16842733)
# ('Epoch: ', 6, 'cost: ', 0.18029341)
# ('Epoch: ', 7, 'cost: ', 0.10681833)
# ('Epoch: ', 8, 'cost: ', 0.32592925)
# ('Epoch: ', 9, 'cost: ', 0.38213861)
# ('Epoch: ', 10, 'cost: ', 0.20461361)
# ('Epoch: ', 11, 'cost: ', 0.22071068)
# ('Epoch: ', 12, 'cost: ', 0.28467309)
# ('Epoch: ', 13, 'cost: ', 0.15771319)
# ('Epoch: ', 14, 'cost: ', 0.11052384)
# ('Epoch: ', 15, 'cost: ', 0.15496463)
# ('Epoch: ', 16, 'cost: ', 0.31195295)
# ('Epoch: ', 17, 'cost: ', 0.16118163)
# ('Epoch: ', 18, 'cost: ', 0.27842495)
# ('Epoch: ', 19, 'cost: ', 0.1902667)
# ('Epoch: ', 20, 'cost: ', 0.077541456)
# ('Epoch: ', 21, 'cost: ', 0.31784946)
# ('Epoch: ', 22, 'cost: ', 0.51690894)
# ('Epoch: ', 23, 'cost: ', 0.25447029)
# ('Epoch: ', 24, 'cost: ', 0.19744799)
# ('Epoch: ', 25, 'cost: ', 0.23692322)
# ('Epoch: ', 26, 'cost: ', 0.067379124)
# ('Epoch: ', 27, 'cost: ', 0.1780508)
# ('Epoch: ', 28, 'cost: ', 0.13570566)
# ('Epoch: ', 29, 'cost: ', 0.21044131)
# 0.962
