# !/usr/bin/python
# -*- coding: utf-8 -*-

from procee_data import get_data,model_train

def MultinomialNB():
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    return model


def SVC():
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', verbose=True)
    return model


def gdbt():
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=300)
    return model


def rfc():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=300)
    return model


def lr():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    return model


# train_vecs,y_train,test_vecs,y_test=get_data()
# svm_train(train_vecs,y_train,test_vecs,y_test)
# string=input('请输入:')
# string, type=model_predict(string)
# print(type+'\n'+ string)
type_model = {'bayes': MultinomialNB(),
              'gdbt': gdbt(),
              'rfc': rfc(),
              'lr': lr(),
              'svm': SVC(),
              }
train_vecs,y_train,test_vecs,y_test=get_data()
for i in type_model:
    model = type_model[i]
    print('model:', model)
    model_train(model, train_vecs, y_train, test_vecs, y_test)

