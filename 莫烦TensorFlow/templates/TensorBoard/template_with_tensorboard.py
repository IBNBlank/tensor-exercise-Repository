# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-11-03 21:44:19
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-11-07 22:49:59

import tensorflow as tf
import numpy as np

###############
### 定义函数 ###
###############
# 添加层
def add_layer(inputs, in_size, out_size, layer_name=None, activation_function=None):
	with tf.name_scope(layer_name):
		with tf.name_scope("Weights"):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
			tf.summary.histogram(layer_name+"/Weights", Weights)
		with tf.name_scope("biases"):
			biases = tf.Variable(tf.zeros([1, out_size])+0.1, name="b")
			tf.summary.histogram(layer_name+"/biases", biases)
		with tf.name_scope("Wx_plus_b"):
			Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)

		tf.summary.histogram(layer_name + '/outputs', outputs)

		return outputs



###############
### 创建数据 ###
###############
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise



################################
### 神经网络构建（一个隐藏层） ###
################################
### 样本 ###
with tf.name_scope("inputs"):
	xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
	ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

### 网络 ###
# 隐藏层
l_1 = add_layer(xs, 1, 10, layer_name="hidden_layer_1", activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(l_1, 10, 1, layer_name="output_layer", activation_function=None)

### 损失函数 ###
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
	tf.summary.scalar("loss", loss)

### 优化器 ###
with tf.name_scope("train"):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)



###############
### 训练模型 ###
###############
### 初始化 ###
# 创建会话
sess = tf.Session()
# Tensorboard 初始化
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/", sess.graph)
# 模型初始化
init = tf.global_variables_initializer()
sess.run(init)

### 训练 ###
for step in range(5001):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if not step % 50:
		result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
		writer.add_summary(result, step)