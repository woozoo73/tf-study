import tensorflow as tf

print "Hello, TensorFlow"

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

y = tf.mul(a, b)

sess = tf.Session()

print sess.run(y, feed_dict={a: 3, b: 2})
