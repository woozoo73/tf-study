import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def showPlot(x_data, y_data, W, b):
    padding = 0.1

    x_min = min(x_data) - padding
    x_max = max(x_data) + padding
    y_min = min(y_data) - padding
    y_max = max(y_data) + padding

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    plt.plot(x_data, y_data, "ro")
    plt.plot(x_data, W * x_data + b)
    plt.show()
    plt.close(1)

def solve(num_points, learning_rate, count, period):
    vectors_set = []

    for i in xrange(num_points):
        x1 = np.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.01)
        vectors_set.append([x1, y1])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]

    W = tf.Variable(tf.random_uniform([1], 10.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    cost = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost);

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in xrange(count):
        sess.run(train)
        if step % period == 0:
            print step, sess.run(W), sess.run(b), sess.run(cost)
            showPlot(x_data, y_data, sess.run(W), sess.run(b))

solve(50, 0.01, 2000, 400)
