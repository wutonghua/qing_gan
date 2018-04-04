#!/usr/bin/python
# -*- coding: utf-8 -*-
from classifier import TextClassifier
from sklearn.externals import joblib
from process_data import get_data
def MultinomialNB():
	from sklearn.naive_bayes import MultinomialNB
	model=MultinomialNB()
	return model
def SVC():
	from sklearn.svm import SVC
	model=SVC(kernel='rbf',verbose=True)
	return model
def gdbt():
	from sklearn.ensemble import GradientBoostingClassifier
	model=GradientBoostingClassifier(n_estimators=300)
	return model
def rfc():
	from sklearn.ensemble import RandomForestClassifier
	model=RandomForestClassifier(n_estimators=300)
	return model
def lr():
	from sklearn.linear_model import LogisticRegression
	model=LogisticRegression()
	return model

x_train,y_train,x_test,y_test=get_data()
type_model={
'lr':lr(),
}
# type_model={'bayes':MultinomialNB(),
# 'gdbt':gdbt(),
# 'rfc':rfc(),
# 'svm':SVC(),
# 'lr':lr(),
# }

for i in type_model:

	model=type_model[i]
	print('model:',model)
	#训练数据
	text_classifier=TextClassifier(model)
	text_classifier.fit(x_train,y_train)
	#保存并加载模型
	joblib.dump(text_classifier, 'text_classifier.pkl')
	print(text_classifier.score(x_test, y_test))
	print("-------------------")