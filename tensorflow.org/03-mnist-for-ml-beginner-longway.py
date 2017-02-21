import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
mnist = input_data.read_data_sets("MNIST_data")

print("len(mnist.train.images)", len(mnist.train.images), "len(mnist.train.labels)", len(mnist.train.labels))
print("len(mnist.test.images)", len(mnist.test.images), "len(mnist.test.labels)", len(mnist.test.labels))

# strategy

# epochs = [1, 2, 4, 10]
# sizes = [1, 100, 200, 1000, 55000]
# rates = [1., 0.5, 0.1, 0.01, 0.001]

epochs = [1, 2, 4]
sizes = [100, 200]
rates = [1., 0.5, 0.01]

for epoch in epochs:
    for size in sizes:
        for rate in rates:
            sess = tf.InteractiveSession()

            x = tf.placeholder(tf.float32, name="x")
            y_ = tf.placeholder(tf.float32, name="y_")
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))

            y = tf.matmul(x, W) + b
            #   (?, 784) * (784 * 10) + (10, ?)
            # = (? * 10)              + (10, ?) # matmul
            # = (? * 10)              + (?, 10) # broadcasting

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train = optimizer.minimize(cross_entropy)

            sess.run(tf.global_variables_initializer())

            total_step = 0
            elapsed_time = None

            start = time.time()
            for c in range(epoch):
                index = 0
                for step in range(len(mnist.train.images) / size):
                    images = mnist.train.images[index * size:(index + 1) * size]
                    labels = mnist.train.labels[index * size:(index + 1) * size]
                    ym = []
                    for v in labels:
                        ymu = [0] * 10  # ymu = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        ymu[v] = 1
                        ym.append(ymu)
                    data = {x: images, y_: ym, learning_rate:rate}
                    train.run(feed_dict=data)
                    total_step = step + 1
                    index += 1
            elapsed_time = time.time() - start

            # predict

            xm = mnist.test.images
            data = {x: xm}

            predicts = sess.run(y, {x: xm})
            # print("predicts.shape", predicts.shape)
            # print("predicts", predicts)

            reals = mnist.test.labels

            success = 0
            failure = 0
            accuracy = 0.
            for i, predict in enumerate(predicts):
                max_val = None
                max_idx = -1
                for j, v in enumerate(predict):
                    if (max_idx == -1) or (max_val < v):
                        max_val = v
                        max_idx = j
                if max_idx == reals[i]:
                    success += 1
                else:
                    failure += 1
                # print(reals[i], max_idx)

            accuracy = (1.0 * success) / (success + failure)

            print([epoch, size, rate, total_step, elapsed_time, success, failure, accuracy])
