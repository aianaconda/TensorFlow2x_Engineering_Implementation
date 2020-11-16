# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""

import tensorflow as tf			# 引入基础模块
import tensorflow.keras
import numpy as np
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
import random

class MyLayer(Layer):
    # 自定义一个类，继承自Layer
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    # 定义build方法用来创建权重
    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # 定义可训练变量
        self.weight = self.add_weight(name='weight',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    # 实现父类的call方法,实现层功能逻辑
    def call(self, inputs):
        return tf.matmul(inputs, self.weight)

    # 如果层更改了输入张量的形状，需要定义形状变化的逻辑
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 单元测试程序
if __name__ == '__main__':
    # 生成训练数据 y=2x
    x_train = np.linspace(0, 10, 100)  		# 100个数
    y_train_random = -1 + 2 * np.random.random(100)	# -1到1之间的随机数
    y_train = 2 * x_train + y_train_random  	# y=2*x + 随机数
    print("x_train \n", x_train)
    print("y_train \n", y_train)

    # 生成测试数据
    x_test = np.linspace(0, 10, 100)  		# 100个数
    y_test_random = -1 + 2 * np.random.random(100)
    y_test = 2 * x_test + y_test_random  		# y=2*x + 随机数
    print("x_test \n", x_test)
    print("y_test \n", y_test)

    # 预测数据
    x_predict = random.sample(range(0, 10), 10)  	# 10个数

    # 定义网络层，一个输入层，三个全连接层
    inputs = Input(shape=(1,))  			# 定义输入张量
    x = Dense(64, activation='relu')(inputs)  	# 第1个全连接层
    x = MyLayer(64)(x)  				# 第2个全连接层，是自定义的层
    predictions = Dense(1)(x)  			# 第3个全连接层

    # 编译模型，指定训练的参数
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',  		# 定义优化器
                  loss='mse',  			# 定义损失函数,绝对误差均值
                  metrics=['mae'])  			# 定义度量

    # 训练模型，指定训练超参数
    history = model.fit(x_train,
                        y_train,
                        epochs=100,  			# 50个epochs
                        batch_size=16)  		# 训练的每批数据量

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
