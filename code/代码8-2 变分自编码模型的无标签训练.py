# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
 
from  tensorflow.keras.layers import *
from  tensorflow.keras.models import *
import tensorflow as tf
import tensorflow.keras.backend as K
from  tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import metrics
# tf.compat.v1.disable_v2_behavior()

batch_size = 100
original_dim = 784   #28*28
latent_dim = 2
intermediate_dim = 256
nb_epoch = 50


class Encoder(tf.keras.Model):  # 提取图片特征
    def __init__(self ,intermediate_dim,latent_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.hidden_layer = Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.z_mean = Dense(units=latent_dim)
        self.z_log_var = Dense(units=latent_dim)

    def call(self, x):
        activation = self.hidden_layer(x)
        z_mean = self.z_mean(activation)
        z_log_var= self.z_log_var(activation)
        return z_mean,z_log_var


def sampling(args): 
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                               stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class Decoder(tf.keras.Model):  # 提取图片特征
    def __init__(self ,intermediate_dim,original_dim,**kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.hidden_layer = Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer  = Dense(units=original_dim, activation='sigmoid')

    def call(self, z):
        activation = self.hidden_layer(z)
        output_layer = self.output_layer(activation)
        return output_layer



encoder = Encoder(intermediate_dim,latent_dim)
decoder = Decoder(intermediate_dim,original_dim)

inputs = Input(batch_shape=(batch_size, original_dim))
z_mean,z_log_var = encoder(inputs)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
y_pred = decoder(z)
autoencoder = Model(inputs, y_pred, name='autoencoder')

# 重构loss
xent_loss = original_dim * metrics.binary_crossentropy(inputs, y_pred)
# KL loss
kl_loss = - 0.5 * K.sum(1 + z_log_var -
                        K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

#########################################y
autoencoder.add_loss(vae_loss)
autoencoder.compile(optimizer='rmsprop')
############################no
# autoencoder.compile(optimizer='rmsprop', loss=vae_loss)
#############################################

 
# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.pkl.gz')
 
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
 
################################################
# autoencoder.fit(x_train, x_train,
#         shuffle=True,
#         epochs=5,#nb_epoch,
#         verbose=2,
#         batch_size=batch_size,
#         validation_data=(x_test, x_test))
 
autoencoder.fit(x_train, validation_split=0.05, epochs=5, batch_size=batch_size)
#################################################


for i in autoencoder.layers:
    print(i)
    
print(autoencoder.layers[1])    #Encoder
print(autoencoder.layers[-1])    #Decoder

# build a model to project inputs on the latent space
modencoder = Model(inputs, z_mean)
 

print(modencoder.layers[1])    #Encoder

# display a 2D plot of the digit classes in the latent space
x_test_encoded = modencoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
