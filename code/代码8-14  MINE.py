# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""
from  tensorflow.keras.layers import *
from  tensorflow.keras.models import *
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

batch_size = 1000  #定义批次大小
#生成模拟数据
def train_generator():
    while(True):
        x = np.sign(np.random.normal(0.,1.,[batch_size,1]))
        y = x+np.random.normal(0.,0.5,[batch_size,1])
        y_shuffle=np.random.permutation(y)
        yield ((x,y,y_shuffle),None)
        
#可视化
for inputs in train_generator():
    x_sample=inputs[0][0]
    y_sample=inputs[0][1]
    plt.scatter(np.arange(len(x_sample)), x_sample, s=10,c='b',marker='o')  
    plt.scatter(np.arange(len(y_sample)), y_sample, s=10,c='y',marker='o')
    plt.show()  
    break


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Dense(10)
        self.fc2 = Dense(10)
        self.fc3 = Dense(1)

    def call(self, x, y):
        # x, y = inputs[0],inputs[1]
        h1 = tf.nn.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2    

model = Net()
optimizer = tf.keras.optimizers.Adam(lr=0.01)#定义优化器

inputs_x = Input(batch_shape=(batch_size, 1))
inputs_y = Input(batch_shape=(batch_size, 1))
inputs_yshuffle = Input(batch_shape=(batch_size, 1))
pred_xy = model(inputs_x,inputs_y)
pred_x_y = model(inputs_x,inputs_yshuffle)


# loss
loss =  -(K.mean(pred_xy) - K.log(K.mean(K.exp(pred_x_y))))  


modeMINE = Model([inputs_x,inputs_y,inputs_yshuffle], [pred_xy,pred_x_y,loss], name='modeMINE')
modeMINE.add_loss(loss)
modeMINE.compile(optimizer=optimizer)
modeMINE.summary()



n = 100
H = modeMINE.fit(x=train_generator(), epochs=n,steps_per_epoch=40,
                     validation_data=train_generator(),validation_steps=4)
    
    
plot_y = np.array(H.history["loss"]).reshape(-1,)
plt.plot(np.arange(len(plot_y)), -plot_y, 'r') 








