# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-10-30 23:37:45
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-10-30 23:41:11

import tensorflow as tf

# Create a Constant op
hello = tf.constant("Hello World")

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))