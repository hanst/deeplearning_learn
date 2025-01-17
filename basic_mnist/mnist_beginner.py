# coding=utf-8
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder("float", shape=[None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W)+b)

y_ = tf.placeholder("float", shape=[None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x:batch_x, y_:batch_y})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("accuracy",sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
