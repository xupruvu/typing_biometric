# -*- coding: utf-8 -*-
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import time

def readDate(latencyFile, pressureFile):
    latencyData = pd.read_csv(latencyFile, sep="\t", header=None)
    latencyData.columns = ['latency_1', 'latency_2','latency_3', 'latency_4','latency_5', 'latency_6','latency_7']
    pressureData = pd.read_csv(pressureFile, sep="\t", header=None)
    pressureData = pressureData.agg(['mean', 'min', 'max', 'median', 'std', 'count'])

    data = latencyData.join(pressureData.transpose())
    data = data.values

    return data

def gen_data(dataPath):
    seed = 20
    userFileList = os.listdir(dataPath)
    userNumber = len(userFileList)
    for i in range(0, userNumber):
        FilePath = os.path.join(dataPath, userFileList[i])
        latencyFile = os.path.join(FilePath, "latency.txt")
        pressureFile = os.path.join(FilePath, "pressure.txt")
        feature_data = readDate(latencyFile, pressureFile)
        label_data = [int(i)]*(feature_data.shape[0])
        X_train_slice, X_test_slice, y_train_slice, y_test_slice = train_test_split(feature_data, label_data, test_size=0.2, random_state=seed)
    
        if i == 0:
            X_train = X_train_slice 
            X_test = X_test_slice 
            y_train = y_train_slice
            y_test = y_test_slice
        else:
            X_train = np.append(X_train,X_train_slice, axis=0)
            X_test = np.append(X_test,X_test_slice, axis=0)
            y_train = np.append(y_train,y_train_slice, axis=0)
            y_test = np.append(y_test,y_test_slice, axis=0)

    train_indices = np.random.permutation(X_train.shape[0])
    test_indices = np.random.permutation(X_test.shape[0])
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    return X_train, y_train, X_test, y_test


def lgb_model(X_train, y_train, X_test, y_test):
    clf = lgb.LGBMClassifier(boosting_type='gbdt',objective='multiclass',metric='auc_mu',
          learning_rate=0.1,
          num_leaves=30, 
          max_depth=10   )
    clf.fit(X_train, y_train)
    # predict the results
    y_pred=clf.predict(X_test)
    # view accuracy
    accuracy=accuracy_score(y_pred, y_test)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    y_pred_train = clf.predict(X_train)
    print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
    # print the scores on training and test set
    print('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    ### https://www.kaggle.com/prashant111/lightgbm-classifier-in-python

def xgb_model(X_train, y_train, X_test, y_test):
    clf = xgb.XGBClassifier(max_depth=5, objective='multi:softmax', n_estimators=1000, num_classes=100)
    clf.fit(X_train, y_train)
    # predict the results
    y_pred=clf.predict(X_test)
    # view accuracy
    accuracy=accuracy_score(y_pred, y_test)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    y_pred_train = clf.predict(X_train)
    print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
    # print the scores on training and test set
    print('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    ### https://blog.csdn.net/u011630575/article/details/79418138
if __name__ == '__main__':
    dataPath = "./keystroke100"
    X_train, y_train, X_test, y_test = gen_data(dataPath)
    lgb_model(X_train, y_train, X_test, y_test)
    #xgb_model(X_train, y_train, X_test, y_test)

