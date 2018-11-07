# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-11-06 23:59:53
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-11-08 01:47:48

import tensorflow as tf
import input_data

###############
### 定义函数 ###
###############
# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
		Weights = tf.Variable(tf.random_normal([in_size, out_size]))
		biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
		Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)

		return outputs

# 检验函数
def compute_accuracy(v_xs, v_ys):
	global prediction

	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(v_ys, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
	print(result)
	return result



###############
### 创建数据 ###
###############
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



################################
### 神经网络构建（一个隐藏层） ###
################################
### 样本 ###
xs = tf.placeholder(tf.float32, [None, 784]) # 28*28
ys = tf.placeholder(tf.float32, [None, 10])

### 网络 ###
# 隐藏层
# l_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 输出层
# prediction = add_layer(l_1, 10, 1, activation_function=None)
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

### 损失函数 ###
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=[1]))
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=prediction), axis=0))

### 优化器 ###
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)


###############
### 训练模型 ###
###############
### 初始化 ###
# 创建会话
sess = tf.Session()
# 模型初始化
init = tf.global_variables_initializer()
sess.run(init)

### 训练 ###
for step in range(10001):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
	if not step % 50:
		print(compute_accuracy(mnist.test.images, mnist.test.labels))