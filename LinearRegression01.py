# origin : https://www.youtube.com/watch?v=mQGwjrStQgg

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal(shape=[1],dtype=tf.float32))
b = tf.Variable(tf.random_normal(shape=[1],dtype=tf.float32))

hypothesis = W * x_data + b
cost = tf.reduce_mean(tf.square(hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10):
    sess.run(train)
    print(step, sess.run(cost), sess.run(W), sess.run(b))