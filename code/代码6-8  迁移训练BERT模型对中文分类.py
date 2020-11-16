# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""

import tensorflow as tf
from transformers import *
import os


data_dir='./THUCNews/data' #定义数据集根目录
#获取分类信息
class_list = [x.strip() for x in open(
        os.path.join(data_dir, "class.txt")).readlines()]

tokenizer = BertTokenizer.from_pretrained(r'./bert-base-chinese/bert-base-chinese-vocab.txt')
#定义指定分类的配置文件
config = AutoConfig.from_pretrained(
        r'./bert-base-chinese/bert-base-chinese-config.json',num_labels=len(class_list)) 
#初始化模型，单独指定config，在config中指定分类个数
model = TFAutoModelForSequenceClassification.from_pretrained(
       r'./bert-base-chinese/bert-base-chinese-tf_model.h5',
        config=config)


def read_file(path): #读取数据集文件内容
    with open(path, 'r', encoding="UTF-8") as file:
        docus = file.readlines()
        newDocus = []
        labs = []
        for data in docus:
            content, label = data.split('\t')
            label = int(label)
            newDocus.append(content)
            labs.append(label)
            
    ids = tokenizer.batch_encode_plus( newDocus,
                #！！！！！模型的配置文件中就是512，当有超过这个长度的会报错
                max_length=model.config.max_position_embeddings,  
                pad_to_max_length=True)#,return_tensors='tf')#没有return_tensors会返回list！！！！
  
    return (ids["input_ids"],ids["attention_mask"],labs)

#获得训练集和测试集
trainContent = read_file(os.path.join(data_dir, "train.txt")) 
testContent = read_file(os.path.join(data_dir, "test.txt"))


def getdataset(features): #定义函数，封装数据集
    
    def gen():              #定义生成器
        for ex in zip(features[0],features[1],features[2]):
            yield (
                {
                    "input_ids": ex[0],
                    "attention_mask": ex[1],
                },
                ex[2],
            )  
      
    return tf.data.Dataset.from_generator( #返回数据集
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                (
                    {
                        "input_ids": tf.TensorShape([None]),
                        "attention_mask": tf.TensorShape([None]),
                    },
                    tf.TensorShape([]),
                ),
            ) 
 
#制作数据集    
valid_dataset = getdataset(testContent) 
train_dataset = getdataset(trainContent) 
#设置批次
train_dataset = train_dataset.shuffle(100).batch(8).repeat(2)
valid_dataset = valid_dataset.batch(16)

#定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

#训练模型
history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

#保存模型
savedir = r'./myfinetun-bert_chinese/'
os.makedirs(savedir, exist_ok=True)
model.save_pretrained(savedir)











