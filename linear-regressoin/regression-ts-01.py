import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x_data = np.random.normal(0.0, 0.9, 3)
y_data = []
for x in x_data:
    y = 0.7 * x + 4 + np.random.normal(0.0, 0.4)
    y_data.append(y)

W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))

hypothesis = W * x_data + b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

print x_data
print y_data
print hypothesis

plt.plot(x_data, y_data, "ro")
plt.show()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in xrange(5001):
    sess.run(train)
    if step % 1000 == 0:
        print step, sess.run(cost), sess.run(W), sess.run(b)
        plt.plot(x_data, y_data, "ro")
        plt.plot(x_data, sess.run(hypothesis), "bo")
        plt.show()
