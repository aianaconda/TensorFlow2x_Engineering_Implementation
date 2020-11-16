# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""


import tensorflow as tf
from transformers import *

# 加载预训练模型（权重）
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


#编码输入
indexed_tokens = tokenizer.encode("Who is Li Jinhong ? Li Jinhong is a")

print( tokenizer.decode(indexed_tokens))

tokens_tensor = tf.constant([indexed_tokens])#转换为张量

# 加载预训练模型（权重）
model = GPT2LMHeadModel.from_pretrained('gpt2')





# 预测所有标记

outputs = model(tokens_tensor)
predictions = outputs[0]#(1, 13, 50257)

# 得到预测的下一词
predicted_index = tf.argmax(predictions[0, -1, :])
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
print(predicted_text)


#生成一段完整的话
stopids = tokenizer.convert_tokens_to_ids(["."])[0] 
past = None
for i in range(100):

    output, past = model(tokens_tensor, past=past)
    token = tf.argmax(output[..., -1, :],axis= -1)

    indexed_tokens += token.numpy().tolist()

    if stopids== token.numpy()[0]:
        break
    tokens_tensor = token[None,:] #增加一个维度
    
sequence = tokenizer.decode(indexed_tokens)

print(sequence)


