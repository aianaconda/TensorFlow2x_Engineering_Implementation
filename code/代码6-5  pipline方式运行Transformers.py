# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""

#代码6-5  pipline方式运行Transformers




################分类##############################

from transformers import *
nlp_sentence_classif= pipeline("sentiment-analysis") 	#自动加载模型
print(nlp_sentence_classif ("I like this book!"))		#调用模型进行处理

################抽取特征##############################
import numpy as np
nlp_features = pipeline('feature-extraction')
output = nlp_features(
           'Code Doctor Studio is a Chinese company based in BeiJing.')
print(np.array(output).shape)   #输出特征形状

################完型填空##############################
nlp_fill = pipeline("fill-mask")
print(nlp_fill.tokenizer.mask_token) #输出遮蔽字符：'[MASK]'
#调用模型进行处理
print(nlp_fill(f"Li Jinhong wrote many {nlp_fill.tokenizer.mask_token} about artificial intelligence technology and helped many people."))	


################阅读理解##############################
nlp_qa = pipeline("question-answering") 		#实例化模型
print(										#输出模型处理结果
  nlp_qa(context='Code Doctor Studio is a Chinese company based in BeiJing.',
           question='Where is Code Doctor Studio?') )


################摘要生成##############################
TEXT_TO_SUMMARIZE = '''
In this notebook we will be using the transformer model, first introduced in this paper. Specifically, we will be using the BERT (Bidirectional Encoder Representations from Transformers) model from this paper.
Transformer models are considerably larger than anything else covered in these tutorials. As such we are going to use the transformers library to get pre-trained transformers and use them as our embedding layers. We will freeze (not train) the transformer and only train the remainder of the model which learns from the representations produced by the transformer. In this case we will be using a multi-layer bi-directional GRU, however any model can learn from these representations.
'''
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small",
                     framework="tf")
print(summarizer(TEXT_TO_SUMMARIZE,min_length=5, max_length=150))




################实体词##############################
nlp_token_class = pipeline("ner")
print(nlp_token_class(
        'Code Doctor Studio is a Chinese company based in BeiJing.'))


from transformers import ALL_PRETRAINED_MODEL_ARCHIVE_MAP
print(ALL_PRETRAINED_MODEL_ARCHIVE_MAP)





