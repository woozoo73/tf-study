import tensorflow as tf

a = [[1, 2], [1, 1]]
b = [[3, 0], [3, 7]]
c = [1, 2, 3]

d = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 100],
     [2, 2, 2, 2, 2, 2, 2, 2, 2, 200],
     [3, 0, 0, 0, 0, 0, 0, 0, 0, 300]]
e = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
f = [[100], [200], [300]]
g = [[3], [3], [3]]
h = 300
i = [300, 200]

sess = tf.Session()

# print sess.run(tf.matmul(a, b))

print sess.run(tf.add(d, e))
print sess.run(tf.add(d, f))
print sess.run(tf.add(d, h))
# print sess.run(tf.add(d, i))

# Reduce Tensor

x = 9

print("-- 0D Tensor ----------------------")
print(sess.run(tf.reduce_sum(x)))
print(sess.run(tf.reduce_mean(x)))

x = [1., 2., 3.]

print("-- 1D Tensor ----------------------")
print(sess.run(tf.reduce_sum(x)))
print(sess.run(tf.reduce_mean(x)))
print(sess.run(tf.reduce_sum(x, axis=0)))
print(sess.run(tf.reduce_mean(x, axis=0)))

x = [
    [1., 2., 3.],
    [1., 1., 1.],
]

print("-- 2D Tensor ----------------------")
print(sess.run(tf.reduce_sum(x)))
print(sess.run(tf.reduce_mean(x)))
print(sess.run(tf.reduce_sum(x, axis=0)))
print(sess.run(tf.reduce_mean(x, axis=0)))
print(sess.run(tf.reduce_sum(x, axis=1)))
print(sess.run(tf.reduce_mean(x, axis=1)))

x = [
    [
        [9., 9., 0.],
        [2., 7., 2.],
    ],
    [
        [0., 0., 0.],
        [1., 1., 1.],
    ],
]

print("-- 3D Tensor ----------------------")
print(sess.run(tf.reduce_sum(x)))
print(sess.run(tf.reduce_mean(x)))
print(sess.run(tf.reduce_sum(x, axis=0)))
print(sess.run(tf.reduce_mean(x, axis=0)))
print(sess.run(tf.reduce_sum(x, axis=1)))
print(sess.run(tf.reduce_mean(x, axis=1)))
print(sess.run(tf.reduce_sum(x, axis=2)))
print(sess.run(tf.reduce_mean(x, axis=2)))

print("-- Elapsed time -------------------")

x = [
    [2, 1, 1.0, 55000, 85.80895805358887, 8909, 1091, 0.8909],
    [2, 1, 0.5, 55000, 90.89890313148499, 8733, 1267, 0.8733],
    [2, 1, 0.1, 55000, 83.18629598617554, 8695, 1305, 0.8695],
    [2, 1, 0.01, 55000, 82.44007992744446, 9104, 896, 0.9104],
    [2, 1, 0.001, 55000, 81.41339182853699, 9090, 910, 0.909],
    [2, 100, 1.0, 550, 2.7207489013671875, 9032, 968, 0.9032],
    [2, 100, 0.5, 550, 2.7411088943481445, 9098, 902, 0.9098],
    [2, 100, 0.1, 550, 2.7443130016326904, 9091, 909, 0.9091],
    [2, 100, 0.01, 550, 2.79909610748291, 8729, 1271, 0.8729],
    [2, 100, 0.001, 550, 2.79502010345459, 7956, 2044, 0.7956],
    [2, 200, 1.0, 275, 2.178550958633423, 9039, 961, 0.9039],
    [2, 200, 0.5, 275, 2.21687388420105, 9121, 879, 0.9121],
    [2, 200, 0.1, 275, 2.3447799682617188, 9025, 975, 0.9025],
    [2, 200, 0.01, 275, 2.1953248977661133, 8532, 1468, 0.8532],
    [2, 200, 0.001, 275, 2.2245070934295654, 7609, 2391, 0.7609],
    [2, 1000, 1.0, 55, 1.503798007965088, 9104, 896, 0.9104],
    [2, 1000, 0.5, 55, 1.4070429801940918, 9021, 979, 0.9021],
    [2, 1000, 0.1, 55, 1.4090209007263184, 8726, 1274, 0.8726],
    [2, 1000, 0.01, 55, 1.4340529441833496, 7956, 2044, 0.7956],
    [2, 1000, 0.001, 55, 1.4185941219329834, 7022, 2978, 0.7022],
    [2, 55000, 1.0, 1, 1.759720802307129, 6051, 3949, 0.6051],
    [2, 55000, 0.5, 1, 1.350200891494751, 7902, 2098, 0.7902],
    [2, 55000, 0.1, 1, 1.25923490524292, 7044, 2956, 0.7044],
    [2, 55000, 0.01, 1, 1.5048999786376953, 6743, 3257, 0.6743],
    [2, 55000, 0.001, 1, 1.4993550777435303, 6711, 3289, 0.6711]
]

w = [[0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.]]

print(sess.run(tf.matmul(x, w)))
print(sess.run(tf.reduce_mean(tf.matmul(x, w), axis=0)))
print(sess.run(tf.reduce_mean(tf.reduce_mean(tf.matmul(x, w), axis=0), axis=0)))
print(sess.run(tf.reduce_mean(tf.matmul(x, w))))


# Reshape Tensor

print("-- Reshape Tensor -----------------")

x = [
    [1, 2, 3, 4, 5, 6],
    [2, 4, 8, 8, 0, 2],
    [4, 8, 12, 4, 4, 4],
    [0, 1, 0, 0, 2, 0],
]

print("-- Reshape Tensor [-1]-------------")

print(sess.run(tf.reshape(x, [-1])))
print(sess.run(tf.reshape(x, [-1, 3, 2])))
# print(sess.run(tf.reshape(x, [-1, 3, -1])))
print(sess.run(tf.reshape(x, [-1, 3, 2, 1])))


print("-- Reduce mean [-1]----------------")

print(x)
print(sess.run(tf.argmax(x, axis=0)))
print(sess.run(tf.argmax(x, axis=1)))
print(sess.run(tf.argmax(x, dimension=1)))
