#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import jieba
import yaml
from keras.models import model_from_yaml
from keras.preprocessing import sequence
maxlen = 50
padding_token=0
#加载模型
print ('loading model......')
with open('./model/lstm.yml', 'r') as f:
    yaml_string = yaml.load(f)
model = model_from_yaml(yaml_string)

print ('loading weights......')
model.load_weights('./model/lstm.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])

#加载词典
dict = pd.read_csv('data/dict.csv')
dict.columns = ["word", "number", "id"]
dict.set_index(['word'], inplace=True)
def input_transform(string):
    word_list=list(jieba.cut(string))
    word_list = [[dict['id'][word]]for word in word_list]
    word_list=list(sequence.pad_sequences(word_list, maxlen=maxlen))
    return word_list

def lstm_predict(string):

    data=np.array(list(input_transform(string)))
    data.reshape(1,-1)
    result=model.predict_classes(data)[0][0]
    if result==1:
        type='positive'
    else:
        type='negtive'
    return type
if __name__=='__main__':
    while True:
        string=input('请输入句子:')
        type=lstm_predict(string)
        print(type)