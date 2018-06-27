# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455

rng = np.random.RandomState(seed)
X = rng.rand(32,2) # 返回32行2列的矩阵，表示32组体积和重量的数据集
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X] # 体积+重量>1为合格零件，作为标签复制给Y
print("X:\n", X)
print("Y:\n", Y)

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

loss = tf.reduce_mean(tf.square(y-y_))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1:\n", sess.run(w1))
    print("w2:\n\n", sess.run(w2)) # 输出当前未训练的参数值

    STEPS = 10000 # 训练10000轮
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d traning step(s), loss on all data is %g" % (i, total_loss))
    print("\nw1:\n", sess.run(w1)) 
    print("w2:\n", sess.run(w2))
