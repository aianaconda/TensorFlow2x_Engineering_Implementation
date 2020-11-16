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


classname = get_attributes(all_class_url,ParsingClass)#获得类名



vcdir= os.path.join(r'./CUBVCfeature/',"ResNet101VC.json") #可见类的VC中心文件json file
#保存可见类的VC中心文件json file
obj=json.load(open(vcdir,"r"))
VC=obj["train"] #获得可见类的中心点

# Obtain the approximated VC of unseen class
vcdir= os.path.join(r'./CUBVCfeature/',"ResNet101VC_testCenter.json") #可见类的VC中心文件json file
obj=json.load(open(vcdir,"r"))
test_center=obj["VC"]


VC = VC+test_center   #源域类别中心点和目的域类别聚类中心点

def NN_search(x,center):
    ret=""
    MINI=-1
    for c in center.keys():
        tmp=np.sum((x-center[c])*(x-center[c]))#L2_dis
#        print(c,tmp)
        if MINI==-1:
            MINI=tmp
            ret=c
        if tmp<MINI:
            MINI=tmp
            ret=c
    return ret


centernpy = np.load("Pred_Center.npy")

center=dict(zip(classname,centernpy))#全部中心点

subcenter = dict(zip(classname[-50:],centernpy[-50:]))#

vcdir= os.path.join(r'./CUBVCfeature/',"ResNet101VC.json") #可见类的VC中心文件json file
#保存可见类的VC中心文件json file
obj=json.load(open(vcdir,"r"))
VC=obj["train"] #获得可见类的中心点
VCunknown = obj["test"]
allVC = VC+VCunknown #视觉中心点
vccenter = dict(zip(classname,allVC))#全部中心点

cur_root = r'./CUBfeature/'
allacc = []

#for target in classname[:classNum-unseenclassnum]: #遍历未知类的特征数据
for target in classname[classNum-unseenclassnum:]: #遍历未知类的特征数据
    cur=os.path.join(cur_root,target)
    fea_name=""
    url=os.path.join(cur,"ResNet101.json")
    js = json.load(open(url, "r"))
    cur_features=js["features"]

    correct=0
    for fea_vec in cur_features:  #### Test the image features of each class
        fea_vec=np.array(fea_vec)
#        ans=NN_search(fea_vec,center)  # Find the nearest neighbour in the feature space
        ans=NN_search(fea_vec,subcenter)
#        ans=NN_search(fea_vec,vccenter)
        
        if ans==target:
            correct+=1

    allacc.append( correct * 1.0 / len(cur_features) )
    print( target,correct)

print("准确率：%.5f"%(sum(allacc)/len(allacc)))



# #测试类别中心点与模型输出的中心点比较
for i,fea_vec in enumerate(VCunknown):  #
    fea_vec=np.array(fea_vec)
    ans=NN_search(fea_vec,center)  # 
    if classname[150+i]!=ans:
        print(classname[150+i],ans)    

# #聚类效果
result = {}
for i,fea_vec in enumerate(test_center):  #### Test the image features of each class
    fea_vec=np.array(fea_vec)
    ans=NN_search(fea_vec,vccenter)  # Find the nearest neighbour in the feature space
    classindex = int(ans.split('.')[0])
    if classindex<=150:
        print("聚类错误的类别",i,ans)
    if classindex not in result.keys():
        result[classindex]=i
    else:
        print("聚类重复的类别",i,result[classindex],ans)
for i in range(150,200):
    if i+1 not in result.keys():
        print("聚类失败的类别：",classname[i])   

