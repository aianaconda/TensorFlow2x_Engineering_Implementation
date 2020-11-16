# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""


import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.activations import *
from tensorflow.keras import optimizers

# (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.pkl.gz')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

batch_size = 100
original_dim = 784  # 28*28




class Encoder(tf.keras.Model):  # 提取图片特征
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.c0 = Conv2D(64, 3, strides=1, activation=tf.nn.relu)  # 26
        self.c1 = Conv2D(128, 3, strides=1, activation=tf.nn.relu)  # 24
        self.c2 = Conv2D(256, 3, strides=1, activation=tf.nn.relu)  # 22
        self.c3 = Conv2D(512, 3, strides=1, activation=tf.nn.relu)  # 20
        self.l1 = Dense(64)  # (512*20*20, 64)

        self.b1 = BatchNormalization()
        self.b2 = BatchNormalization()
        self.b3 = BatchNormalization()

    def call(self, x):
        x = Reshape((28, 28, 1))(x)
        h = self.c0(x)

        features = self.b1(self.c1(h))  # b 24 24 128
        h = self.b2(self.c2(features))
        h = self.b3(self.c3(h))

        h = Flatten()(h)
        encoded = self.l1(h)  # b 64
        return encoded, features


class DeepInfoMaxLoss(tf.keras.Model):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1, **kwargs):
        super(DeepInfoMaxLoss, self).__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.encoder = Encoder()

        self.local_d = Sequential([
            Conv2D(512, 1, strides=1, activation=tf.nn.relu),
            Conv2D(512, 1, strides=1, activation=tf.nn.relu),
            Conv2D(1, 1, strides=1)
        ])

        self.prior_d = Sequential([
            Dense(1000, batch_input_shape=(None, 64), activation=tf.nn.relu),
            Dense(200, activation=tf.nn.relu),
            Dense(1, activation=tf.nn.sigmoid),
        ])

        self.global_d_M = Sequential([
            Conv2D(64, 3, activation=tf.nn.relu),  # b 64 22 22
            Conv2D(32, 3),  # b 64 20 20
            Flatten()
        ])

        self.global_d_fc = Sequential([
            Dense(512, activation=tf.nn.relu),
            Dense(512, activation=tf.nn.relu),
            Dense(1),
        ])

    def call(self, x):
        y, M = self.encoder(x)
        return self.thiscall(y, M)

    def thiscall(self, y, M):
        M_prime = tf.concat([M[1:], tf.expand_dims(M[0], 0)], 0)  # (b,24, 24, 128)

        y_exp = Reshape((1, 1, 64))(y)  # b 1,1,64
        y_exp = tf.tile(y_exp, [1, 24, 24, 1])  # b 24,24 64,
        
        y_M = tf.concat((M, y_exp), -1)  # b 24,24 192
        y_M_prime = tf.concat((M_prime, y_exp), -1)  # b 24,24 192
       

        Ej = -K.mean(softplus(-self.LocalD(y_M)))
        Em = K.mean(softplus(self.LocalD(y_M_prime)))
        LOCAL = (Em - Ej) * self.beta

        Ej = -K.mean(softplus(-self.GlobalD(y, M)))
        Em = K.mean(softplus(self.GlobalD(y, M_prime)))
        GLOBAL = (Em - Ej) * self.alpha
        
        prior = K.random_uniform(shape=(K.shape(y)[0], K.shape(y)[1]))

        term_a = K.mean(K.log(self.PriorD(prior)))
        term_b = K.mean(K.log(1.0 - self.PriorD(y)))
        PRIOR = - (term_a + term_b) * self.gamma

        return GLOBAL, LOCAL, PRIOR  
    
    def LocalD(self, x):
        return self.local_d(x)

    def PriorD(self, x):
        return self.prior_d(x)

    def GlobalD(self, y, M):
        h = self.global_d_M(M)
        h = tf.concat((y, h), -1)
        return self.global_d_fc(h)


dimer = DeepInfoMaxLoss()

# 定义优化器
optimizer = optimizers.Adam(lr=0.0001)

import os

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")  # 定义检查点文件的路径
# 定义检查点文件
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=dimer)
latest_cpkt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_cpkt:  # 处理二次训练
    print('Using latest checkpoint at ' + latest_cpkt)
    checkpoint.restore(latest_cpkt)
else:
    os.makedirs(checkpoint_dir, exist_ok=True)  # 建立存放模型的文件夹

# K.set_learning_phase(True)
dimer.trainable = True
training_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
nb_epoch = 1
for epoch in range(nb_epoch):  # 按照指定迭代次数进行训练
    for dataone in training_dataset:  # 遍历数据集
        img = np.reshape(dataone, (batch_size, -1))
        with tf.GradientTape() as tape:
            GLOBAL, LOCAL, PRIOR = dimer(img)
            if PRIOR < 0.00001:
                thisloss = LOCAL + GLOBAL
            else:
                thisloss = LOCAL + GLOBAL + PRIOR
            print("thisloss:", thisloss.numpy(), GLOBAL.numpy(), LOCAL.numpy(), PRIOR.numpy())
            gradients = tape.gradient(thisloss, dimer.trainable_variables)
            gradient_variables = zip(gradients, dimer.trainable_variables)
            optimizer.apply_gradients(gradient_variables)
    print(epoch, " loss:", thisloss)



checkpoint.save(file_prefix=checkpoint_prefix)
#####################################################

inputs = Input(batch_shape=(None, original_dim))
y, M = dimer.encoder(inputs)
modeENCODER = Model(inputs, y, name='modeENCODER')
modeENCODER.save_weights('my_model.h5')


modeENCODER.load_weights('my_model.h5')
testn = 5000
x_test_encoded = modeENCODER.predict(np.reshape(x_test[:testn], (len(x_test[:testn]), -1)), batch_size=batch_size)

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['font.family'] = 'STSong'
mpl.rcParams['font.size'] = 20

try:
    # https://www.deeplearn.me/2137.html  perplexity困惑度 n_components最终维度
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=2000)  # 5000
    low_dim_embs = tsne.fit_transform(x_test_encoded)
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], c=y_test[:testn])
plt.colorbar()
plt.show()



# 验证搜索
import random

index = random.randrange(0, len(y_test[:testn]))  # 随机获取一个索引
mse = list(map(lambda x: ((x_test_encoded[index] - x) ** 2).sum(), x_test_encoded))

user_ranking_idx = sorted(enumerate(mse), key=lambda p: p[1])  # , reverse=True

findy = [y_test[i] for i, v in user_ranking_idx]
print(y_test[index], findy[:20])
