"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""
import glob
import json
import os
import re
import shutil
import tensorflow as tf
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np

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

from tensorflow.keras.applications import ResNet101V2
batchsize = 1

cur_root = 'E:\\Project\\08-TF\\TF2\\ZSL_TF2\\CUBfeature\\'
VC_dir = 'E:\\Project\\08-TF\\TF2\\ZSL_TF2\\CUBVCfeature\\'
os.makedirs(cur_root, exist_ok=True)#创建目录，用于存放视觉特征
dataset_path = "E:\\Project\\08-TF\\TF2\\Caltech-UCSD-Birds-200-2011\\CUB_200_2011\\images\\"
     
image_model = ResNet101V2(weights='resnet.h5',include_top=False, pooling = 'avg')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output #获取ResNet的倒数第二层（池化前的卷积结果）

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

size = [224,224]
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)#(shape is (224,224,3))
    return img, image_path

def avgfeature(all_features):#计算类别的平均特征
    #对类别中所有图片的特征求平均值
    avg_features = np.sum(np.asarray(all_features),axis = 0)/len(all_features)
    avg_features = tf.convert_to_tensor(avg_features)
    avg_features = tf.linalg.normalize(avg_features)[0]#对平均特征归一化
    avg_features = avg_features.numpy()#形状为(2048,)
    return avg_features

def savefeature(classdir,filename,obj): #定义函数保存特征
    os.makedirs(classdir, exist_ok=True)#创建子目录
    cur_url = os.path.join(classdir,filename)#定义文件名称
    json.dump(obj,open(cur_url,"w"))#保存json文件
    print("%s has finished ..."%(classdir))

(filenames,labels),_ =load_sample(dataset_path,shuffleflag=False) #载入文件名称与标签,length of filenames is 8822
'''
filenames and labels are the same as they are in pytorch
'''
image_dataset = tf.data.Dataset.from_tensor_slices(filenames).map(load_image).batch(batchsize)
ith_features = []
all_features = []
target_VC = []
test_VC = []
classsplit = 150
number = 1
for img, path in tqdm(image_dataset):
    '''
    shape of img:(1,224,224,3)
    '''
    batch_features = image_features_extract_model(img)
    batch_features = tf.squeeze(batch_features)
    batch_features = tf.linalg.normalize(batch_features)[0]
    path_of_feature = path[0].numpy().decode("utf-8")
    path_of_feature = path_of_feature[path_of_feature.index('images\\')+len("images\\"):]
    path_of_feature = path_of_feature.replace('\\','_')
    current_number = int(path_of_feature[:3])
    if current_number == number:
        ith_features.append(batch_features.numpy().tolist())
        all_features.append(batch_features)
    else:
        classdir= os.path.join(cur_root,"%03d"%number) #定义第i类的路径
        obj={}
        obj["features"]=ith_features
        savefeature(classdir,"ResNet101.json",obj)
        avg_feature = avgfeature(all_features)
        if number < classsplit+1:
            target_VC.append(avg_feature.tolist())
        else:
            test_VC.append(avg_feature.tolist())
        all_features.clear()    
        ith_features.clear()
        number += 1
        continue
        
classdir= os.path.join(cur_root,str(number)) #定义第i类的路径
obj={}
obj["features"]=ith_features
savefeature(classdir,"ResNet101.json",obj)

avg_feature = avgfeature(all_features)
test_VC.append(avg_feature.tolist())
obj = {}
obj["train"] = target_VC
obj["test"] = test_VC
savefeature(VC_dir,"ResNet101VC.json",obj)

alltestfeatures = []
feature_json_list = os.listdir(cur_root)
feature_json_list.sort()
for dirpath in feature_json_list[classsplit:]:
    cur=os.path.join(cur_root,dirpath)
    fea_name=""
    url=os.path.join(cur,"ResNet101.json")
    js = json.load(open(url, "r"))
    cur_features=js["features"]
    alltestfeatures =alltestfeatures+cur_features
      
from sklearn.cluster import KMeans

def KM(features):
    clf = KMeans(n_clusters=len(feature_json_list)-classsplit, 
                 n_init=50, max_iter=100000, init="k-means++")

    print("Start Cluster ...")
    s=clf.fit(features)
    print("Finish Cluster ...")

    obj={}
    obj["VC"]=clf.cluster_centers_.tolist()

    print('Start writing ...')
    savefeature(VC_dir,"ResNet101VC_testCenter.json",obj)
    print("Finish writing ...")
    
KM(alltestfeatures)


