# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-11-03 18:42:27
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-11-04 21:30:56

import tensorflow as tf
import numpy as np

###############
### 创建数据 ###
###############
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

###########################
### 创建 TensorFlow 结构 ###
###########################
# 权重 和 偏置
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
# 模型
y = Weights*x_data + biases
# 损失函数
loss = tf.reduce_mean(tf.square(y-y_data))
# 优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

###############
### 训练模型 ###
###############
# 结构初始化
init = tf.initialize_all_variables()
# 创建会话
sess = tf.Session()
sess.run(init) # 激活（很重要，不要忘）
# 训练
for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(Weights), sess.run(biases))