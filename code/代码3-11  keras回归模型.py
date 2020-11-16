# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""

import numpy as np				# 引入基础模块
import random
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# 生成训练数据 y=2x+随机数
x_train = np.linspace(0, 10, 100)  		# 100个数
y_train_random = -1 + 2 * np.random.random(100)  	# -1到1之间的随机数
y_train = 2 * x_train + y_train_random  		# y=2*x +随机数
print("x_train \n", x_train)
print("y_train \n", y_train)

# 生成测试数据
x_test = np.linspace(0, 10, 100)  			# 100个数
y_test_random = -1 + 2 * np.random.random(100)  	# -1到1之间的随机数
y_test = 2 * x_test + y_test_random  		# y=2*x +随机数
print("x_test \n", x_test)
print("y_test \n", y_test)

# 预测数据
x_predict = random.sample(range(0, 10), 10)  	# 10个数

# 定义网络层，一个输入层，三个全连接层
inputs = Input(shape=(1,))  			# 定义输入张量
x = Dense(64, activation='relu')(inputs)  		# 第一个全连接层
x = Dense(64, activation='relu')(x)  		# 第二个全连接层
predictions = Dense(1)(x)  			# 第三个全连接层，

# 编译模型，指定训练的参数
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',  		# 定义优化器
              loss='mse',  				# 损失函数是均方差
              metrics=['mae'])  			# 定义度量,绝对误差均值

# 训练模型，指定训练超参数
history = model.fit(x_train,
                    y_train,
                    epochs=100,  			# 100个epochs
                    batch_size=16)  			# 训练的每批数据量

# 测试模型
score = model.evaluate(x_test,
                       y_test,
                       batch_size=16)  		# 测试的每批数据量
# 打印误差值和评估标准值
print("score \n", score)

# 模型预测
y_predict = model.predict(x_predict)
print("x_predict \n", x_predict)
print("y_predict \n", y_predict)
