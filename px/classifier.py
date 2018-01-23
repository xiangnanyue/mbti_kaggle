# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = xgb.XGBRegressor(max_depth=10, learning_rate=0.5, n_estimators=100, 
                                     silent=True, booster='gbtree',
                                     eval_metric='auc',
                                     n_jobs=1, nthread=None, gamma=0, min_child_weight=1, 
                                     max_delta_step=0, subsample=1, colsample_bytree=1, 
                                     colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                                     scale_pos_weight=1, base_score=0.5, random_state=0, 
                                     seed=None, missing=None)

    def fit(self, X, y):
        self.clf.fit(X, list(map(int, y)))

    def predict(self, X):
        pred = self.clf.predict(X)
        result = np.zeros((X.shape[0], 6))
        for i in range(X.shape[0]):
            idx = int(round(pred[i]))
            if(idx > 5):
                idx = 5
            result[i][idx] = 1
        return result

    def predict_proba(self, X):
        pred = self.clf.predict(X)
        result = np.zeros((X.shape[0], 6))
        for i in range(X.shape[0]):
            idx = int(round(pred[i]))
            if(idx > 5):
                idx = 5
            result[i][idx] = 1
        return result