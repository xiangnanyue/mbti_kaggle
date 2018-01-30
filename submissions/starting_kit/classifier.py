# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = XGBClassifier(n_estimators=88, learning_rate=0.05)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        y = self.clf.predict_proba(X)
        return y
