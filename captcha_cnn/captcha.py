# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_HEIGHT = 114
IMAGE_WIDTH = 450
MAX_CAPTCHA = 6
CHAR_SET_LEN = 26


#获取图片和名称
def get_name_and_image():
    image_list = os.listdir("captcha4/")
    random_file = random.randint(0, 3429)
    name = os.path.splitext(image_list[random_file])[0]
    image = np.array(Image.open('captcha4/' + image_list[random_file]))
    return name, image


#名称转向量
def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * 26 + ord(c) - 97
        vector[idx] = 1
    return vector


#向量转名称
def vec2name(vec):
    name = []
    for i in vec:
        a = chr(i + 97)
        name.append(a)
    return "".join(name)


#生成一个训练batch
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1 * (image.flatten())  #flatten将二维数组一维化
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)


#定义cnn
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    #3个卷积层
    #5,5分别表示卷积核的高度和宽度 1表示图像通道数 32表示卷积核数
    w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]), name="w_c1")
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]), name="b_c1")
    conv1 = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(
        conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  #池化层
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]), name="w_c2")
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]), name="b_c2")
    conv2 = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'),
            b_c2))
    conv2 = tf.nn.max_pool(
        conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([5, 5, 64, 64]), name="w_c3")
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]), name="b_c3")
    conv3 = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'),
            b_c3))
    conv3 = tf.nn.max_pool(
        conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    #全连接层
    w_d = tf.Variable(
        w_alpha * tf.random_normal([15 * 57 * 64, 1024]), name="w_d")
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]), name="b_d")
    dense = tf.reshape(conv3, [-1, 15 * 57 * 64])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    #输出层
    w_o = tf.Variable(
        w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]),
        name="w_o")
    b_o = tf.Variable(
        b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]), name="b_o")
    out = tf.add(tf.matmul(dense, w_o), b_o)
    return out


def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_l, max_idx_p)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1000
        saver.restore(sess, "./crack_capcha.ckpt-999")
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run(
                [optimizer, loss],
                feed_dict={
                    X: batch_x,
                    Y: batch_y,
                    keep_prob: 0.5
                })
            print("step:{} loss_:{}".format(step, loss_))

            if step % 100 == 0:  #每100步测试一次正确率
                batch_x_test, batch_y_test = get_next_batch(64)
                acc = sess.run(
                    accuracy,
                    feed_dict={
                        X: batch_x_test,
                        Y: batch_y_test,
                        keep_prob: 1.
                    })
                print("step:{} acc:{}".format(step, acc))

                if acc > 0.99:
                    saver.save(sess, "./crack_capcha.ckpt", global_step=step)
                    break
            if (step + 1) % 500 == 0:  #每500步存储一次
                saver.save(sess, "./crack_capcha.ckpt", global_step=step)
            step += 1


train_crack_captcha_cnn()