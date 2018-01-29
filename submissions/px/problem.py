# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from datetime import timedelta

problem_title = 'personality prediction'
_target_column_name = 'type'
_prediction_label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
dict = {'INFJ':0, 'ENTP':1, 'INTP':2, 'INTJ':3, 'ENTJ':4, 'ENFJ':5, 'INFP':6, 'ENFP':7, 'ISFP':8, 'ISTP':9,
 'ISFJ':10, 'ISTJ':11, 'ESTP':12, 'ESFP':13, 'ESTJ':14, 'ESFJ':15}
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

soft_score_matrix = np.identity(16)

for i in dict:
    for j in dict:
        n = 0
        for k in j:
            if(k in i): n+=1
        soft_score_matrix[dict[i]][dict[j]] = n / 4.
        

true_false_score_matrix = np.identity(16)


        
score_types = [
    rw.score_types.SoftAccuracy(
        name='sacc', score_matrix=soft_score_matrix, precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
    rw.score_types.SoftAccuracy(
        name='tfacc', score_matrix=true_false_score_matrix, precision=3),
]


def get_cv(X, y):
    """Slice folds by equal date intervals."""
#     date = pd.to_datetime(X['date'])
#     n_days = (date.max() - date.min()).days
    n_splits = 8
    fold_length = X.shape[0]/n_splits
    arr = np.array(list(range(X.shape[0])))
    np.random.shuffle(arr)
    for i in range(n_splits):
        test = arr[int(i*fold_length):int((i+1)*fold_length)]
        train = np.concatenate((arr[:int(i*fold_length)],arr[int((i+1)*fold_length):]))
        yield(arr[train], arr[test])
        
def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), sep=',')
    y_array1 = data[_target_column_name].values
    y_array = np.array([dict[k] for k in y_array1])
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[:100], y_array[:100]
    else:
        return X_df[:1000], y_array[:1000]


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


