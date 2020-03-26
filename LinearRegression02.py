# origin - https://www.youtube.com/watch?v=mQGwjrStQgg

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = tf.placeholder(dtype=tf.float32, shape=[None])
y_data = tf.placeholder(dtype=tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = W * x_data + b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    c_, w_, b_, _ = sess.run([cost, W, b, train],
         feed_dict={x_data : [1,2,3,4,5], y_data : [2.1,3.1,4.1,5.1,6.1]})
    if( step % 20 == 0 ):
        print(step, c_, w_, b_)
