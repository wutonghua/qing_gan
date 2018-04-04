#!/usr/bin/python
# -*- coding: utf-8 -*-
from procee_data import model_predict
while True:
	string=input('请输入:')
	string, type=model_predict(string)
	print(type+'\n'+ string)