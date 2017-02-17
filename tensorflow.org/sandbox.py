import tensorflow as tf

a = [[1, 2], [1, 1]]
b = [[3, 0], [3, 7]]
c = [1, 2, 3]

d = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 100],
     [2, 2, 2, 2, 2, 2, 2, 2, 2, 200],
     [3, 0, 0, 0, 0, 0, 0, 0, 0, 300]]
e = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
f = [[100],[200],[300]]
g = [[3], [3], [3]]
h = 300
i = [300, 200]

sess = tf.Session()

# print sess.run(tf.matmul(a, b))

print sess.run(tf.add(d, e))
print sess.run(tf.add(d, f))
print sess.run(tf.add(d, h))
print sess.run(tf.add(d, i))
