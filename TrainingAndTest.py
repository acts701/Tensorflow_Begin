# origin - https://www.youtube.com/watch?v=oSJfejG2C3w&feature=youtu.be

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x_test = [[2,1,1],[3,1,2],[3,3,4]]
y_test = [[0,0,1],[0,0,1],[0,0,1]]

X = tf.placeholder(tf.float32, [None,3])
Y = tf.placeholder(tf.float32, [None,3])

W = tf.Variable(tf.random_normal(shape=[3,3],dtype=tf.float32))
b = tf.Variable(tf.random_normal(shape=[3],dtype=tf.float32))

# 이 모델을 사용하면 accuracy가 85%
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predict = tf.cast(hypothesis > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        _, _cost, _W = sess.run([optimizer, cost, W], feed_dict={X:x_data,Y:y_data})
        if step % 100 == 0:
            print('cost = ', _cost)

    print(' pre = ', sess.run(predict, feed_dict={X:x_test, Y:y_test}))
    print('accu = ', sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
    # print(' pre = ', sess.run(predict, feed_dict={X:x_data, Y:y_data}))
    # print('accu = ', sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
