#!/usr/bin/python
# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import random
from load_dataset import preprocess_text
from sklearn.model_selection import train_test_split
def get_data():

	df_pos = pd.read_excel('data/pos.xls',header=None, index=None)
	df_pos = df_pos.dropna()
	df_neg = pd.read_excel('data/neg.xls',header=None, index=None)
	df_neg = df_neg.dropna()

	pos = df_pos[0].values.tolist()
	neg = df_neg[0].values.tolist()

	#生成训练数据
	sentences = []
	preprocess_text(pos, sentences, 'pos')
	preprocess_text(neg, sentences, 'neg')
	random.shuffle(sentences)
	x, y = zip(*sentences)
	train_data, test_data, train_target, test_target = train_test_split(x, y, random_state=1234)
	return train_data, test_data, train_target, test_target