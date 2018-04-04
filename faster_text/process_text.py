#!/usr/bin/python
# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import random
from load_dataset import preprocess_text
cate_dic = {'pos':1, 'neg':2}


df_pos = pd.read_excel('data/pos.xls',header=None, index=None)
df_pos = df_pos.dropna()
df_neg = pd.read_excel('data/neg.xls',header=None, index=None)
df_neg = df_neg.dropna()

pos = df_pos[0].values.tolist()
neg = df_neg[0].values.tolist()

#生成训练数据
sentences = []
preprocess_text(pos, sentences, cate_dic['pos'])
preprocess_text(neg, sentences, cate_dic['neg'])
random.shuffle(sentences)
print("writing data to fasttext format...")
out = open('data/train_data.txt', 'w',encoding='utf-8')
for sentence in sentences:
    out.write(sentence+"\n")
print ("done!")