import tensorflow as tf

w1 = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
ema_op = ema.apply(tf.trainable_variables())
# ema.apply 后的括号里是更新列表，每次运行 sess.run(ema_op) 时，对更新列表中的参数求滑动平均值。

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([w1, ema.average(w1)]))
    # 打印 w1 和 w1 的滑动平均值
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

