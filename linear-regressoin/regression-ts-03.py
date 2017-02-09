import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.normal(5.0, 0.9, 100)
y_data = []
for x in x_data:
    y = 3 * x + 4 + np.random.normal(0.0, 0.7);

plt.plot(x_data, y_data)
plt.show()

W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))

hyposis = W * x_data + b
cost = tf.reduce_mean(tf.square(y_data - hyposis))

print W, b, hyposis

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in xrange(1000):
    print step, sess.run(W), sess.run(b), sess.run(hyposis), sess.run(cost)
