# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-11-03 21:22:58
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-11-03 21:31:32

import tensorflow as tf

###############
### 创建数据 ###
###############
one = tf.constant(1)

###########################
### 创建 TensorFlow 结构 ###
###########################
state = tf.Variable(0, name="counter")
# print(state.name)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

######################
### Variable 的使用 ###
######################
# 初始化
init = tf.initialize_all_variables()
# 运行
with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))