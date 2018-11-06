# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-11-03 21:44:19
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-11-04 22:11:35

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###############
### 定义函数 ###
###############
# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases

	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)

	return outputs

###############
### 创建数据 ###
###############
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

################################
### 神经网络构建（一个隐藏层） ###
################################
# 样本
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 隐藏层
l_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(l_1, 10, 1, activation_function=None)
# 损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

###############
### 训练模型 ###
###############
# 可视化初始化
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
# 创建会话
sess = tf.Session()
# 模型初始化
init = tf.initialize_all_variables()
sess.run(init)
# 训练
for step in range(1001):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if step % 50:
		# print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		prediction_value = sess.run(prediction, feed_dict={xs: x_data})
		lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
		plt.pause(0.1)
# 关闭可视化
print("Over")
plt.pause(15)
plt.close()