# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-11-03 21:32:46
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-11-03 21:38:35

import tensorflow as tf

###############
### 创建数据 ###
###############
input_1 = tf.placeholder(tf.float32)
input_2 = tf.placeholder(tf.float32)

###########################
### 创建 TensorFlow 结构 ###
###########################
output = tf.multiply(input_1, input_2)

#########################
### placeholder 的使用 ###
#########################
with tf.Session() as sess:
	print(sess.run(output, feed_dict={input_1:[7.], input_2:[2.]}))