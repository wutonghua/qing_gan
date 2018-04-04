#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split as sp
from sklearn import metrics
from procee_data import get_data
from sklearn.externals import joblib
svc=SVC(probability=True)
gnb=GNB()
knn=KNN()
rfc=RFC(random_state=1)
model=[
	svc,gnb,knn
]
#加载数据
X_train,y_train,X_test,y_test=get_data()
#训练模型
X_train_d1,X_train_d2,y_train_d1,y_train_d2=sp(X_train,y_train,test_size=0.5,random_state=1)
X_train_d2_blending=np.zeros((X_train_d2.shape[0],len(model)))
X_test_blending=np.zeros((X_test.shape[0],len(model)))
# X_train_d1=X_train_d1.todense()
for j,clf in enumerate(model):
	print(clf)
	clf.fit(X_train_d1,y_train_d1)
	y_test_value=clf.predict_proba(X_train_d2)[:,1]
	X_train_d2_blending[:,j]=y_test_value
	X_test_blending[:,j]=clf.predict_proba(X_test)[:,1]
	print('测试集AUC是: {:.4}'.format(metrics.roc_auc_score(y_test,X_test_blending[:,j])))
	print('------------------------------------')
lr=LR()
lr.fit(X_train_d2_blending,y_train_d2)
joblib.dump(lr, 'svm_data/w2v_model/lr_model.pkl')
y_test_value=lr.predict_proba(X_test_blending)[:,1]
y_test_value=(y_test_value - y_test_value.min()) / (y_test_value.max() - y_test_value.min())
print('融合后，测试集AUC是： {:.4}'.format(metrics.roc_auc_score(y_test,y_test_value)))
