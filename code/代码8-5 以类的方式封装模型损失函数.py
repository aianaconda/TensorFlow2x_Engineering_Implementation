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


#import sys
#saveout = sys.stdout
#file = open('variational_autoencoder.txt','w')
#sys.stdout = file
# 
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


#encoder.summary()


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



class VAE(tf.keras.Model):  # 提取图片特征
    def __init__(self ,intermediate_dim,original_dim,latent_dim,**kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = Encoder(intermediate_dim,latent_dim)
        self.decoder = Decoder(intermediate_dim,original_dim)

    def call(self, x):
        z_mean,z_log_var = self.encoder(x)
        z = sampling( (z_mean,z_log_var) )
        y_pred = self.decoder(z)
        xent_loss = original_dim * metrics.binary_crossentropy(x, y_pred) #ok
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        loss = xent_loss + kl_loss
        return loss


#encoder = Encoder(intermediate_dim,latent_dim)
#decoder = Decoder(intermediate_dim,original_dim)

vae = VAE(intermediate_dim,original_dim,latent_dim)


#inputs = Input(batch_shape=(batch_size, original_dim))
#vae_lossmode = vae(inputs)
#lossautoencoder = Model(inputs, vae_lossmode, name='lossautoencoder')
#lossautoencoder.add_loss(vae_lossmode)
#lossautoencoder.compile(optimizer='rmsprop')


def vae_loss(x, loss):
    return loss
vae.compile(optimizer='rmsprop', loss=vae_loss)

 
# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.pkl.gz')
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
 
#lossautoencoder.fit(x_train, validation_split=0.05, epochs=5, batch_size=batch_size)
#vae.fit(x_train, x_train,
#        shuffle=True,
#        epochs=5,#nb_epoch,
#        verbose=2,
#        batch_size=batch_size,
#        validation_data=(x_test, x_test))
optimizer = Adam(lr=0.001) 
training_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
nb_epoch = 5
for epoch in range(nb_epoch):  	# 按照指定迭代次数进行训练
    for dataone in training_dataset:  	# 遍历数据集
        img = np.reshape(dataone, (batch_size, -1))
        with tf.GradientTape() as tape:
            thisloss = K.mean( vae(img) )

            gradients = tape.gradient(thisloss, vae.trainable_variables)
            gradient_variables = zip(gradients, vae.trainable_variables)
            optimizer.apply_gradients(gradient_variables)
    print(epoch," loss:",thisloss.numpy())






inputs = Input(batch_shape=(batch_size, original_dim))
z_mean,z_log_var = vae.encoder(inputs)

# build a model to project inputs on the latent space
modencoder = Model(inputs, z_mean)
 

print(modencoder.layers[1])    #Encoder

# display a 2D plot of the digit classes in the latent space
#x_test_encoded = modencoder.predict(x_test, batch_size=batch_size)
x_test_encoded,_ = vae.encoder(x_test)



plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
