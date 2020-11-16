# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""
import json
import numpy as np
import os
import tensorflow as tf

SinkhornDistance = __import__("代码9-1  Sinkhorn")



class FC(tf.keras.Model):
    '''
    定义全连接模型将属性特征从312维映射至2048维度
    '''
    def __init__(self):
        super(FC, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
         
        self.dense1 = tf.keras.layers.Dense(1024, input_shape=(1,), name = 'dense1')
        self.dense2 = tf.keras.layers.Dense(2048, name = 'dense2')

        self.outputlayer = tf.keras.layers.Dense(2048, name = 'outputlayer')
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor):#, training=True
        x = self.dense1(input_tensor)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.outputlayer(x)
        return x

def WDVSc(x,y,epsilon,niter,no_use_VSC=False):
    '''
    定义损失函数
    '''
    sum_ = 0
    for  mask_idx in range(150):
        sum_ += (x[mask_idx] - y[mask_idx]) ** 2
    L2_loss = tf.reduce_sum(sum_) / (150 * 2) ## L2_Loss of seen classes
    A = x[150:]
    B = y[150:]
    if no_use_VSC:
        WD_loss=0.
        P=None
        C=None
    else:
        WD_loss,P,C = SinkhornDistance.sinkhorn_loss(A,B,epsilon,niter,reduction = 'sum')
    lamda=0.001
    tot_loss=L2_loss+WD_loss*lamda
    return tot_loss


def ParsingAtt(lines):#312列依次转成浮点
    line=lines.strip().split()
    cur=[]
    for x in line:
        y=float(x)
        y=y/100.0
        if y<0.0:
            y=0.0
        cur.append(y)
    return cur

def ParsingClass(lines):#类名，取第二项
    line=lines.strip().split()
    return line[1]

def get_attributes(url,function=None):
    data=[]
    with open(url,"r") as f:
        for lines in f:
            cur =function(lines)
            data.append(cur)
    return data


input_dim=312
classNum=200
unseenclassnum=50
data_path=r'D:\样本\图片\Caltech-UCSD Birds-200-2011\Caltech-UCSD Birds-200-2011'
attributes_url = os.path.join(data_path,"CUB_200_2011/attributes/class_attribute_labels_continuous.txt")
all_class_url = os.path.join(data_path, "CUB_200_2011/classes.txt")

att = get_attributes(attributes_url,ParsingAtt) #获得属性
classname = get_attributes(all_class_url,ParsingClass)#获得类名

word_vectors = tf.convert_to_tensor(att)#(200,312)
word_vectors = tf.linalg.normalize(word_vectors)[0]#(200,312)

vcdir= os.path.join(r'./CUBVCfeature/',"ResNet101VC.json") #可见类的VC中心文件json file
#保存可见类的VC中心文件json file
obj=json.load(open(vcdir,"r"))
VC=obj["train"] #获得可见类的中心点

# Obtain the approximated VC of unseen class
vcdir= os.path.join(r'./CUBVCfeature/',"ResNet101VC_testCenter.json") #可见类的VC中心文件json file
obj=json.load(open(vcdir,"r"))
test_center=obj["VC"]


VC = VC+test_center   #源域类别中心点和目的域类别聚类中心点
vc_tensor = tf.convert_to_tensor(VC)#加载平均视觉特征

print("vc_tensor,word_vectors",tf.shape(vc_tensor),tf.shape(word_vectors))




batch_size = 1
lr=0.001
wd=0.0005
max_epoch=5000



optimizer = tf.keras.optimizers.Adam(learning_rate=lr) 

Net = FC()

loss_list = []


################################################  

for i in range(max_epoch):
        
    with tf.GradientTape() as tape:
        tape.watch(Net.variables)
        syn_vc = Net(word_vectors)
        loss = WDVSc(vc_tensor,syn_vc,epsilon = 0.01,  niter = 1000)
    grad=tape.gradient(target=loss,sources=Net.variables)
    optimizer.apply_gradients(zip(grad,Net.variables))
    loss_list.append(loss)
    print("{}-th loss = {}".format(i, loss ))
    if i+1 % 1000 == 0:
        checkpoint_path = "check_point\\"
        if os.path.exists(checkpoint_path):
            pass
        else:
            os.mkdir(checkpoint_path)
        Net.save_weights(checkpoint_path+'model_%s.ckpt'%str(i))

print('training is over')
     
 ####################################################


output_vectors = Net(word_vectors)   #调用模型，根据属性生成特征    

np.save("Pred_Center.npy",output_vectors)



