#!/usr/bin/python
# -*- coding: utf-8 -*-
import fasttext
from load_dataset import preprocess
classifier = fasttext.supervised('data/train_data.txt', 'classifier.model', label_prefix='__label__')
# result = classifier.test('train_data.txt')
# print ('P@1:', result.precision)
# print ('R@1:', result.recall)
# print ('Number of examples:', result.nexamples)
label_to_cate = {1:'pos', 2:'neg'}
while True:
	line=input('请输入语句:')
	line=preprocess(line)
	labels = classifier.predict(line)
	print(labels)
	print(label_to_cate[int(labels[0][0])])