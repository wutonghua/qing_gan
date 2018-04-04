#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from load_dataset import preprocess

import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders
from process_text import get_data
learn = tf.contrib.learn

FLAGS = None
MAX_DOCUMENT_LENGTH = 15
MIN_WORD_FREQUENCE = 1
EMBEDDING_SIZE = 50

# 处理词汇
train_data, test_data, train_target, test_target=get_data()
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
x_train = np.array(list(vocab_processor.fit_transform(train_data)))
x_test = np.array(list(vocab_processor.transform(test_data)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)
cate_dic = {'pos':1, 'neg':2}
train_target = map(lambda x:cate_dic[x], train_target)
test_target = map(lambda x:cate_dic[x], test_target)
y_train = pandas.Series(train_target)
y_test = pandas.Series(test_target)
def bag_of_words_model(features, target):
	"""先转成词袋模型"""
	target = tf.one_hot(target, 15, 1, 0)
	features = encoders.bow_encoder(
			features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
	logits = tf.contrib.layers.fully_connected(features, 15, activation_fn=None)
	loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
	train_op = tf.contrib.layers.optimize_loss(
			loss,
			tf.contrib.framework.get_global_step(),
			optimizer='Adam',
			learning_rate=0.01)
	return ({
			'class': tf.argmax(logits, 1),
			'prob': tf.nn.softmax(logits)
	}, loss, train_op)


model_fn = bag_of_words_model
classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn))

# Train and predict
classifier.fit(x_train, y_train, steps=1000)
y_predicted = classifier.predict(x_test)['class']
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: {0:f}'.format(score))
while True:
	line=input('请输入语句:')
	line=preprocess(line)
	line=np.array(list(vocab_processor.transform(line)))
	y_predicted = classifier.predict(line)['class'][0]
	if y_predicted==2:
		lei_bie='neg'
	else:
		lei_bie = 'pos'
	print(lei_bie)