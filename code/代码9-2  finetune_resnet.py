"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""
import glob
import os
import re
import shutil
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()

def load_sample(sample_dir,shuffleflag = True):
    '''递归读取文件。只支持一级。返回文件名、数值标签、数值对应的标签名'''
    print ('loading sample dataset..')
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):#递归遍历文件夹
        for filename in filenames:                            #遍历所有文件名
            #filename:每个文件夹下，每种鸟的名字
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)               #添加文件名
            labelsnames.append( dirpath.split('\\')[-1] )#添加文件名对应的标签

    lab= list(sorted(set(labelsnames)))  #生成标签名称列表
    labdict=dict( zip( lab  ,list(range(len(lab)))  )) #生成字典

    labels = [labdict[i] for i in labelsnames]

    if shuffleflag == True:
        return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)
    else:
        return (np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)
    
def _norm_image(image,size,ch=1,flattenflag = False):    #定义函数，实现归一化，并且拍平
    image_decoded = image/127.5-1#image/255.0
    if flattenflag==True:
        image_decoded = tf.reshape(image_decoded, [size[0]*size[1]*ch])
    return image_decoded
    
def dataset(directory,size,batchsize,shuffleflag = True):#定义函数，创建数据集
    """ parse  dataset."""
    (filenames,labels),_ =load_sample(directory,shuffleflag=False) #载入文件名称与标签
    def _parseone(filename, label):                         #解析一个图片文件
        """ Reading and handle  image"""
        image_string = tf.io.read_file(filename)         #读取整个文件
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape([None, None, None])    # 必须有这句，不然下面会转化失败
        image_decoded = tf.image.resize(image_decoded, size)  #变化尺寸
        image_decoded = _norm_image(image_decoded,size)#归一化
        image_decoded = tf.cast(image_decoded, dtype=tf.float32)
        #label = tf.cast(tf.reshape(label, [1,]) ,tf.int32)#将label 转为张量
        label = tf.one_hot(indices = label, depth = 150)
        return image_decoded, label
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))#生成Dataset对象
    if shuffleflag == True:#乱序
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(_parseone)   #有图片内容的数据集
    dataset = dataset.batch(batchsize) #批次划分数据集
    dataset = dataset.prefetch(1)
    return dataset

from tensorflow.keras.applications import ResNet101V2

def imgs_input_fn(dataset):
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)			#生成一个迭代器
    one_element = iterator.get_next()					#从iterator里取出一个元素
    return one_element

tf.compat.v1.global_variables_initializer()	

size = [224,224]
batchsize = 10

dataset_path = "Caltech-UCSD-Birds-200-2011\\CUB_200_2011\\train\\"
test_path = "Caltech-UCSD-Birds-200-2011\\CUB_200_2011\\test\\"

traindataset = dataset(dataset_path,size,batchsize)#训练集
testdataset = dataset(test_path,size,batchsize,shuffleflag = False)#测试集
next_batch_train = imgs_input_fn(traindataset)				#从traindataset里取出一个元素
next_batch_test = imgs_input_fn(testdataset)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)#(shape is (224,224,3))
    return img, image_path

def showresult(subplot,title,thisimg):          #显示单个图片
    p =plt.subplot(subplot)
    p.axis('off')
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):   #显示
    plt.figure(figsize=(20,10))     #定义显示图片的宽、高
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()

with tf.compat.v1.Session() as sess:	# 建立会话（session）
    sess.run(tf.compat.v1.global_variables_initializer())  #初始化

    try:
        for step in np.arange(1):
            value = sess.run(next_batch_train)
            showimg(step,value[1],np.asarray( (value[0]+1)*127.5,np.uint8),10)       #显示图片


    except tf.errors.OutOfRangeError:           #捕获异常
        print("Done!!!")


img_size = (224, 224, 3)
inputs = tf.keras.Input(shape=img_size)
image_model = ResNet101V2(weights='resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
                          input_tensor=inputs,
                          input_shape = img_size,
                          include_top=False)

model = tf.keras.models.Sequential()
model.add(image_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(150, activation='softmax'))
image_model.trainable = False
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])
#训练模型
model_dir ="./models/test"
os.makedirs(model_dir, exist_ok=True)
print("model_dir: ",model_dir)

est_birds = tf.keras.estimator.model_to_estimator(keras_model=model,  model_dir=model_dir)
#训练模型
import time
start_time = time.time()
with tf.compat.v1.Session() as sess1:	# 建立会话（session）
    sess1.run(tf.compat.v1.global_variables_initializer())  #初始化
    try:
        train__=sess1.run(next_batch_train)
        eval__=sess1.run(next_batch_test)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: train__,max_steps=500)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval__)
        tf.estimator.train_and_evaluate(est_birds, train_spec, eval_spec)
        
    except tf.errors.OutOfRangeError:           #捕获异常
        print("Done!!!")

print("--- %s seconds ---" % (time.time() - start_time))
image_model.save('resnet.h5')
