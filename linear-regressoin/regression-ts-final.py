import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def make_x_data(size):
    xs = np.random.normal(5.0, 0.9, size)
    return xs


def make_y_data(xs):
    ys = []
    for x in xs:
        y = 0.73 * x + 4 + np.random.normal(0.0, 0.2);
        ys.append(y)
    return ys


def draw_plot():
    padding = 1
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(hypothesis), 'b')
    plt.xlim([min(x_data) - padding, max(x_data) + padding])
    plt.ylim([min(y_data) - padding, max(y_data) + padding])
    plt.show()


def draw_cost():
    plt.plot(cx, cy, 'b')
    plt.show()


def solve(learning_rate, count, by):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    for step in xrange(count + 1):
        cx.append(step)
        cy.append(sess.run(cost))

        sess.run(train)
        if step % by == 0:
            print step, sess.run(W), sess.run(b), sess.run(cost)
            draw_plot()


data_size = 100

x_data = make_x_data(data_size)
y_data = make_y_data(x_data)

cx = []
cy = []

W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))

hypothesis = W * x_data + b
cost = tf.reduce_mean(tf.square(y_data - hypothesis))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

solve(0.01, 1000, 100)
draw_cost()
