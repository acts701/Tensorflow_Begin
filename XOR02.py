# origin - https://www.youtube.com/watch?v=GYecDQQwTdI&feature=youtu.be

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

# x의 in이 2개이고 y의 out이 1이니까 shape는 [2,1]
W = tf.Variable(tf.random_normal(shape=[2,1], dtype=tf.float32))
# bias는 out의 개수와 동일
b = tf.Variable(tf.random_normal(shape=[1], dtype=tf.float32))

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        _, _cost, _W = sess.run([train, cost, W], feed_dict={X:x_data, Y:y_data})
        if step % 20 == 0:
            print(step, ' = ', _cost, _W)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\n hypothesis : ", h)
    print("\n Correct : ", c)
    print("\n Accuracy : ", a)
