import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

print("sess.run(node2)", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# adder_node = a * b
adder_node = tf.matmul(a, b)

print(sess.run(adder_node, {a: [[1, 2], [1, 1]], b: [[3, 0], [3, 7]]}))
