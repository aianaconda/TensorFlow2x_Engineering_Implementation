# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""




import tensorflow as tf
from transformers import *

#加载预训练模型 tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')






#输入文本
text = "[CLS] Who is Li Jinhong ? [SEP] Li Jinhong is a programmer [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

masked_index = 15 #掩码一个标记，用' BertForMaskedLM '预测回来
tokenized_text[masked_index] = '[MASK]'
print(tokenized_text)

# 将标记转换为词汇表索引
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# 将输入转换为张量
tokens_tensor = tf.constant([indexed_tokens])


# 加载预训练模型 (weights)
model = TFAutoModelWithLMHead.from_pretrained('bert-base-uncased')


segments_ids = [0, 0, 0, 0, 0, 0, 0, 0,0,0, 1, 1, 1, 1, 1, 1, 1,1,1]
segments_tensors = tf.constant([segments_ids])


# 预测所有的tokens
# output = model(tokens_tensor)
outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    
predictions = outputs[0]  #[1, 19, 30522]

predicted_index = tf.argmax(predictions[0, masked_index])
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0] #转成单词
print('Predicted token is:',predicted_token)


