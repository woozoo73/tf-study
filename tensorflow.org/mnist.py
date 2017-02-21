from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

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

mnist = input_data.read_data_sets("MNIST_data")

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
