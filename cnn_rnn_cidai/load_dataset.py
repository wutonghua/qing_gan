#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import jieba
stopwords=pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values
def load_dataset(name):
    datasets = {
        'neg': 'neg.xls',
        'pos': 'pos.xls',
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('data', datasets[name])
    df = pd.read_excel(data_file,header=None, index=None)
    return df
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            segs = filter(lambda x:len(x)>1, segs)
            segs = filter(lambda x:x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
        except Exception as e:
            print(line)
            continue

def preprocess(line):

    sentence=[]
    segs=jieba.lcut(line)
    segs = filter(lambda x:len(x)>1, segs)
    segs = filter(lambda x:x not in stopwords, segs)
    sentence.append(" ".join(segs))
    return sentence
if __name__=='__main__':
   sentence=preprocess('我来北京了')
   print(sentence)