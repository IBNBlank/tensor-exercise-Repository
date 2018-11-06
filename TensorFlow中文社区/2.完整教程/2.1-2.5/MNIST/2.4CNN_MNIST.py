# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-11-02 00:11:29
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-11-03 18:19:47

import tensorflow as tf
import input_data

###################
##### 前期准备 #####
###################

##### 加载数据 #####
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

##### 运行 InteractiveSession #####
sess = tf.InteractiveSession()

###########################
##### 构建多层卷积网络 #####
###########################

##### 定义函数 #####
### 权重初始化
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
### 偏置初始化
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
### 卷积
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
### 池化
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

##### 定义模型 #####
### 第一层卷积 ###
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
