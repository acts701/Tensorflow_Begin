# origin - https://www.youtube.com/watch?v=oFGHOsAYiz0&feature=youtu.be

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.set_random_seed(777)

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal(shape=[2,10],dtype=tf.float32))
b = tf.Variable(tf.random_normal(shape=[10],dtype=tf.float32))
layer1 = tf.sigmoid(tf.matmul(X, W) + b)

W2 = tf.Variable(tf.random_normal(shape=[10,1],dtype=tf.float32))
b2 = tf.Variable(tf.random_normal(shape=[1],dtype=tf.float32))
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predict = tf.cast(hypothesis > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        _, _cost, _W = sess.run([train, cost, W2], feed_dict={X:x_data,Y:y_data})
        if step % 100 == 0:
            print(step, _cost, _W)

    _hypothesis, _predict, _accuracy = sess.run([hypothesis, predict, accuracy],feed_dict={X:x_data, Y:y_data})
    print("h = ", _hypothesis)
    print('p = ', _predict)
    print('a = ', _accuracy)