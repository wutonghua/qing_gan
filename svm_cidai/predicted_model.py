#!/usr/bin/python
# -*- coding: utf-8 -*-
from process_data import model_predict
from sklearn.externals import joblib
new_text_classifier = joblib.load('text_classifier.pkl')
#输出预测类别
while True:
	line=input('请输入:')
	string, type=model_predict(new_text_classifier,line)
	print(type + '\n' +string)
