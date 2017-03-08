from tensorflow.examples.tutorials.mnist import input_data

print("-------- one_hot=True ---------")

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch = mnist.train.next_batch(2)
print(batch)

batch = mnist.train.next_batch(2)
print(batch)

batch = mnist.train.next_batch(2)
print(batch)

print(batch[0].shape)
print(batch[1].shape)

print(batch[0])
print(batch[1])
print(batch[0][0])
print(batch[1][0])

print(len(mnist.train.images))
print(len(mnist.train.labels))
print(len(mnist.validation.images))
print(len(mnist.validation.labels))
print(len(mnist.test.images))
print(len(mnist.test.labels))

print("-------- one_hot=False --------")

mnist = input_data.read_data_sets("MNIST_data")

batch = mnist.train.next_batch(2)
print(batch)

print(batch[0].shape)
print(batch[1].shape)

print("batch[0]      : %s" % batch[0])
print("batch[1]      : %s" % batch[1])
print("batch[0][0]   : %s" % batch[0][0])
print("batch[1][0]   : %s" % batch[1][0])

print("MNIST train images size     : %d" % len(mnist.train.images))
print("MNIST train labels size     : %d" % len(mnist.train.labels))
print("MNIST validation images size: %d" % len(mnist.validation.images))
print("MNIST validation labels size: %d" % len(mnist.validation.labels))
print("MNIST test images size      : %d" % len(mnist.test.images))
print("MNIST test images size      : %d" % len(mnist.test.images))
