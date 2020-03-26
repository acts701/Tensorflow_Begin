# origin - https://www.youtube.com/watch?v=oFGHOsAYiz0&feature=youtu.be

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal(shape=[2,2],dtype=tf.float32))
b1 = tf.Variable(tf.random_normal(shape=[2],dtype=tf.float32))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal(shape=[2,1],dtype=tf.float32))
b2 = tf.Variable(tf.random_normal(shape=[1],dtype=tf.float32))
hypothesis = tf.sigmoid(tf.matmul(layer1,W2) + b2)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        _, _cost, _W = sess.run([train,cost,W2],feed_dict={X:x_data,Y:y_data})
        if( step % 5 == 0):
            print(step, " = ", _cost, _W )

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\n hypothesis : ", h)
    print("\n Correct : ", c)
    print("\n Accuracy : ", a)
