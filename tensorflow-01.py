import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

mul = tf.mul(a, b)
reciprocal = tf.reciprocal(a)

sess = tf.Session()

print a
print b
print mul

print sess.run(mul, feed_dict={a: 3, b:2})
print sess.run(reciprocal, feed_dict={a: 7})
