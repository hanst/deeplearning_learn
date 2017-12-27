# coding=utf-8

import tensorflow as tf
import numpy as np

input_x = np.float32(np.random.rand(2,100))
y_data = np.dot([0.1,0.2],input_x)+0.3

print(input_x.shape)

b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1,2],-1,1))

print(w)
y = tf.matmul(w, input_x)+b

loss = tf.reduce_mean(tf.square(tf.subtract(y, y_data)))
optimize = tf.train.GradientDescentOptimizer(0.5)

train = optimize.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(0,201):
    sess.run(train)
    if i%20==0:
        print("step", i, "w=",sess.run(w),"b=",sess.run(b))
