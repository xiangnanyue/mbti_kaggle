# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Classifier(BaseEstimator):
    def __init__(self):
        #self.clf = RandomForestClassifier()
        self.clf = SVC(probability=True,class_weight='balanced',kernel='linear')

    def fit(self, X, y):
        #print(y)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        #print(X.shape)
        y = self.clf.predict_proba(X)
        #print(y.shape)
        return y