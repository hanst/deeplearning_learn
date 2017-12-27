# coding=utf-8
import image_input_2
import tensorflow as tf

x = tf.placeholder("float", shape=[None, 1024*1024])
y_ = tf.placeholder("float", shape=[None, 15])
train, test = image_input_2.read_data_sets()
print("data is ready")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_varibale(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=inital)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME" )

W_conv1 = weight_variable([5,5,1,32])
B_conv1 = bias_varibale([32])

x_image = tf.reshape(x, [-1,1024,1024,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+B_conv1)
h_pool1 = max_pool2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
B_conv2 = bias_varibale([64])

hconv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+B_conv2)
h_pool2 = max_pool2x2(hconv2)

W_conv3 = weight_variable([5,5,64,64])
B_conv3 = bias_varibale([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3)+B_conv3)
h_pool3 = max_pool2x2(h_conv3)

W_conv4 = weight_variable([5,5,64,64])
B_conv4 = bias_varibale([64])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4)+B_conv4)
h_pool4 = max_pool2x2(h_conv4)

W_fc1 = weight_variable([64*64*64,1024])
b_fc1 = bias_varibale([1024])

h_pool_plat = tf.reshape(h_pool4, [-1,64*64*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_plat, W_fc1)+b_fc1)

keep_prob1 = tf.placeholder("float")

h_fc1_dropout1 = tf.nn.dropout(h_fc1, keep_prob=keep_prob1)

W_fc2 = weight_variable([1024,15])
B_fc2 = bias_varibale([15])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_dropout1, W_fc2)+B_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y_conv,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print("====begin to train=====")
for i in range(1000):
    batch = train.next_batch(50)
    if i%10 ==0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob1:1.0})
        print("step: %d, training accuracy %d", i, train_accuracy)
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob1:0.5})

print("test accuracy %d", accuracy.eval(feed_dict={
    x: test.images, y_: test.labels, keep_prob1: 1.0}))