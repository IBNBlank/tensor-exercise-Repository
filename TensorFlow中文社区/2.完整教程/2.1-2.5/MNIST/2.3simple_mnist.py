# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-10-20 00:00:46
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-11-02 00:07:43

import tensorflow as tf
import input_data

##### 前期准备 #####
### 加载数据 ###
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

### 运行 InteractiveSession ###
sess = tf.InteractiveSession()

### 模型 ###
### 模型：y = W*x + b
# 占位符
x = tf.placeholder("float", [None,784])
y_ = tf.placeholder("float", [None,10])
# 变量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 回归模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

### 损失函数 ###
# 交叉熵
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))

### 训练方法 ###
# 梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


##### 运行过程 #####
### 初始化变量 ###
sess.run(tf.initialize_all_variables())

### 开始训练 ###
for i in range(1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

### 模型评估 ###
# 正确的预测列表
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))