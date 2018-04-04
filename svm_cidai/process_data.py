#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

def load_file_and_preprocessing():
	neg = pd.read_excel('data/neg.xls', header=None, index=None)
	pos = pd.read_excel('data/pos.xls', header=None, index=None)

	# cw = lambda x: list(jieba.cut(x))
	# pos['words'] = pos[0].apply(cw)
	# neg['words'] = neg[0].apply(cw)
	cw1 = lambda x: ' '.join(list(jieba.cut(x)))
	pos['words'] = pos[0].apply(cw1)
	neg['words'] = neg[0].apply(cw1)

	# print pos['words']
	# use 1 for positive sentiment, 0 for negative
	y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

	x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

	np.save('svm_data/y_train.npy', y_train)
	np.save('svm_data/y_test.npy', y_test)
	np.save('svm_data/x_train.npy', x_train)
	np.save('svm_data/x_test.npy', x_test)
def get_data():
	x_train=np.load('svm_data/x_train.npy')
	y_train=np.load('svm_data/y_train.npy')
	x_test=np.load('svm_data/x_test.npy')
	y_test=np.load('svm_data/y_test.npy')
	return x_train,y_train,x_test,y_test
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
def model_predict(model,string):

	line = model.process_line(string)
	type = model.predict(line)[0]
	lei_bie=''
	if int(type) == 1:
		lei_bie=lei_bie+'positive'

	else:
		lei_bie = lei_bie + 'negtive'
	return string, lei_bie
if __name__=='__main__':
	load_file_and_preprocessing()
