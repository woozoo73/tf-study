import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, name='x')
    y_ = tf.placeholder(tf.float32, name='y_')

with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # tf.summary.image('input', x_image, max_outputs=55000)

with tf.name_scope('weights'):
    W = tf.Variable(tf.zeros([784, 10]), name='W')
    W_784 = tf.transpose(W)
    W_28x28 = tf.reshape(W_784, [-1, 28, 28, 1])
    tf.summary.image('W', W_28x28, max_outputs=100)
    tf.summary.histogram('W', W)

with tf.name_scope('biases'):
    b = tf.Variable(tf.zeros([10]), name='b')
    tf.summary.histogram('b', b)

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
tf.summary.scalar('cross_entropy', cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = None

sess = tf.Session()

for learning_rate in [1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3, 1e+4]:
    sess.run(tf.global_variables_initializer())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cross_entropy)
    train_writer = tf.summary.FileWriter("/tmp/mnist_logs/train/%s" % learning_rate)
    for step in range(2000):
        batch = mnist.train.next_batch(100)
        summary, _ = sess.run([merged, train], {x: batch[0], y_: batch[1]})
        train_writer.add_summary(summary, step)
        summary, _ = sess.run([merged, accuracy], {x: batch[0], y_: batch[1]})
        train_writer.add_summary(summary, step)

    print sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})
