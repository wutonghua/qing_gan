
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
import itertools
from sklearn.externals import joblib
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


def load_file_and_preprocessing():
	neg = pd.read_excel('data/neg.xls', header=None, index=None)
	pos = pd.read_excel('data/pos.xls', header=None, index=None)

	cw = lambda x: list(jieba.cut(x))
	pos['words'] = pos[0].apply(cw)
	neg['words'] = neg[0].apply(cw)

	# print pos['words']
	# use 1 for positive sentiment, 0 for negative
	y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

	x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

	np.save('svm_data/y_train.npy', y_train)
	np.save('svm_data/y_test.npy', y_test)
	return x_train, x_test
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


def get_train_vecs(x_train, x_test):
	n_dim = 300
	# 初始化模型和词表
	imdb_w2v = Word2Vec(size=n_dim, min_count=10)
	imdb_w2v.build_vocab(x_train)

	# 在评论训练集上建模
	imdb_w2v.train(x_train,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)

	train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
	# train_vecs = scale(train_vecs)
	train_vecs = min_max_scaler.fit_transform(train_vecs)
	np.save('svm_data/train_vecs.npy', train_vecs)
	# 在测试集上训练
	imdb_w2v.train(x_test,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)
	imdb_w2v.save('svm_data/w2v_model/w2v_model.pkl')
	# Build test tweet vectors then scale
	test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
	# test_vecs = scale(test_vecs)
	test_vecs = min_max_scaler.fit_transform(test_vecs)
	np.save('svm_data/test_vecs.npy', test_vecs)
def get_data():
	train_vecs=np.load('svm_data/train_vecs.npy')
	y_train=np.load('svm_data/y_train.npy')
	test_vecs=np.load('svm_data/test_vecs.npy')
	y_test=np.load('svm_data/y_test.npy')
	return train_vecs,y_train,test_vecs,y_test
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
	clf = joblib.load('svm_data/w2v_model/model.pkl')
	result = clf.predict(words_vecs)
	type=''
	if int(result[0]) == 1:
		type=type+'positive'

	else:
		type = type + 'negtive'
	return string, type
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
def model_train(clf,train_vecs,y_train,test_vecs,y_test):
	clf=clf
	clf.fit(train_vecs,y_train)
	joblib.dump(clf, 'svm_data/w2v_model/model.pkl')
	print(clf.score(test_vecs,y_test))
	print("--------------------")
if __name__=='__main__':
	x_train, x_test=load_file_and_preprocessing()
	get_train_vecs(x_train, x_test)