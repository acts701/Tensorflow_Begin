# origin : https://www.youtube.com/watch?v=mQGwjrStQgg

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32)

hypothesis = x_train * W + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        sess.run(train)
        print(step, sess.run(cost), sess.run(W), sess.run(b))