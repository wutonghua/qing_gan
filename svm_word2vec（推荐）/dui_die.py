#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  StratifiedKFold
import numpy as np
from procee_data import get_data
from sklearn.externals import joblib
#堆叠模型
clf1 = RandomForestClassifier(n_estimators=100,min_samples_split=5,max_depth=10)
clf2 = SVC()
clf3 = LogisticRegression()
basemodes = [
            ['rf', clf1],
            ['svm', clf2],
            ['lr', clf3]
            ]
cv = StratifiedKFold(n_splits=5)
models = basemodes

train_vecs,y_train,test_vecs,y_test=get_data()
S_train = np.zeros((train_vecs.shape[0], len(models)))
S_test = np.zeros((test_vecs.shape[0], len(models)))

for i, bm in enumerate(models):
	clf = bm[1]

	# S_test_i = np.zeros((y_test.shape[0], len(folds)))
	for train_idx, test_idx in cv.split(train_vecs, y_train):

		X_train_cv = train_vecs[train_idx]
		y_train_cv = y_train[train_idx]
		X_val = train_vecs[test_idx]
		clf.fit(X_train_cv, y_train_cv)
		y_val = clf.predict(X_val)[:]

		S_train[test_idx, i] = y_val
	S_test[:, i] = clf.predict(test_vecs)

final_clf = LogisticRegression()
final_clf.fit(S_train, y_train)
joblib.dump(final_clf, 'svm_data/w2v_model/final_model.pkl')


print(final_clf.score(S_test, y_test))