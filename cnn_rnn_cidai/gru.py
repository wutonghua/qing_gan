#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from process_text import get_data
from load_dataset import preprocess
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
def rnn_model(features, target):
	"""用RNN模型(这里用的是GRU)完成文本分类"""
	# Convert indexes of words into embeddings.
	# This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
	# maps word indexes of the sequence into [batch_size, sequence_length,
	# EMBEDDING_SIZE].
	word_vectors = tf.contrib.layers.embed_sequence(
			features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')

	# Split into list of embedding per word, while removing doc length dim.
	# word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
	word_list = tf.unstack(word_vectors, axis=1)

	# Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
	cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

	# Create an unrolled Recurrent Neural Networks to length of
	# MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
	_, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

	# Given encoding of RNN, take encoding of last step (e.g hidden size of the
	# neural network of last step) and pass it as features for logistic
	# regression over output classes.
	target = tf.one_hot(target, 15, 1, 0)
	logits = tf.contrib.layers.fully_connected(encoding, 15, activation_fn=None)
	loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

	# Create a training op.
	train_op = tf.contrib.layers.optimize_loss(
			loss,
			tf.contrib.framework.get_global_step(),
			optimizer='Adam',
			learning_rate=0.01)

	return ({
			'class': tf.argmax(logits, 1),
			'prob': tf.nn.softmax(logits)
	}, loss, train_op)
model_fn = rnn_model
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