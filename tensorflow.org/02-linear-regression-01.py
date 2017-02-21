import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable([0.3])
b = tf.Variable([-0.3])
h = W * x + b
loss = tf.reduce_mean(tf.square(h - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(1001):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    if step % 100 == 0:
        print(step, sess.run([W, b]))

print(sess.run(h, {x: 2.5}))
print(sess.run(h, {x: [100, 200, 300]}))
