#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba
class TextClassifier():

    def __init__(self,classifier):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,4), max_features=50000)
        # self.vectorizer = TfidfTransformer()
    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)
    def process_line(self,x):
        segs = jieba.lcut(x)
        segs = filter(lambda x: len(x) > 1, segs)
        segs = " ".join(segs)
        return segs