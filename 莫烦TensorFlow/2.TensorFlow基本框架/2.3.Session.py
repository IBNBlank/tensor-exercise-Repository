# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-11-03 19:15:18
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-11-03 20:49:56

import tensorflow as tf

###############
### 创建数据 ###
###############
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[1]])

###########################
### 创建 TensorFlow 结构 ###
###########################
product = tf.matmul(matrix1, matrix2)

#####################
### Session 的使用 ###
#####################
# method 1
sess = tf.Session()
result_1 = sess.run(product)
print(result_1)
sess.close()
# method 2
with tf.Session() as sess:
	result_2 = sess.run(product)
	print(result_2)