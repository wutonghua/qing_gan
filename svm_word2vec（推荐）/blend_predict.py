#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from gensim.models.word2vec import Word2Vec
import jieba
from sklearn.externals import joblib
def build_sentence_vector(text, size,imdb_w2v):
	vec = np.zeros(size).reshape((1, size))
	count = 0.
	for word in text:
		try:
			vec += imdb_w2v[word].reshape((1, size))
			count += 1.
		except KeyError:
			continue
	if count != 0:
		vec /= count
	return vec
def get_predict_vecs(words):
	n_dim = 300
	imdb_w2v = Word2Vec.load('svm_data/w2v_model/w2v_model.pkl')
	#imdb_w2v.train(words)
	train_vecs = build_sentence_vector(words, n_dim,imdb_w2v)
	#print train_vecs.shape
	return train_vecs
def model_predict(string):
	words = jieba.lcut(string)
	words_vecs = get_predict_vecs(words)
	words_vecs=np.array(words_vecs).reshape(-1,3)
	clf = joblib.load('svm_data/w2v_model/lr_model.pkl')
	result = clf.predict(words_vecs)
	percent=sum(result ) / len(result)
	# type=''
	# if int(result[0]) == 1:
	# 	type=type+'positive'
	#
	# else:
	# 	type = type + 'negtive'
	# return string, type
	return percent
while True:
	string=input('请输入:')
	result=model_predict(string)
	# print(type+'\n'+ string)
	print(result)