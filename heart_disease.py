# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:36:25 2022

@author: ChenMingfeng
"""

import pandas as pd

if __name__ == "__main__":
    # 读取csv文件
    path = "C:/Users/ChenMingfeng/Downloads/documents-export-2022-11-27/1.Stroke prediction/Stroke_prediction_system_Data/"
    train_feature_path = path + "Prf_feature_train.csv"
    train_label_path = path + "Stroke_label_train.csv"
    test_feature_path = path + "Prf_feature_test.csv"

    # 删去第一列
    train_features = pd.read_csv(train_feature_path).iloc[:, 1:]
    train_labels = pd.read_csv(train_label_path).iloc[:, 1:]
    test_features = pd.read_csv(test_feature_path).iloc[:, 1:]
    print(train_features.shape)
    print(train_labels.shape)
    print(test_features.shape)
    
    # 数据预处理
    train_fetures = train_features.fillna(method='ffill')
    
    redundant_cols = ['FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'CTELENM1', 'SEQNO', 'CTELNUM1', 'NUMHHOL3', 'NUMPHON3',
                      'CPDEMO1B', 'X_CLLCPWT', 'X_DUALUSE', 'X_DUALCOR', 'X_LLCPWT2', 'X_LLCPWT', 'CELPHON1']
    for c in redundant_cols:
        del train_features[c], test_features[c]
    
    print(train_features.dtypes)
    
    #选择有心脏病的样本
    label = []
    for i in range(train_features.shape[0]):
        if(train_features.loc[i]['HeartDisease'] == 2):
            label.append(i)
    train_features = train_features.drop(labels=label, axis=0)
    print(train_features.shape)
    
    label = []
    for i in range(test_features.shape[0]):
        if(test_features.loc[i]['HeartDisease'] == 2):
            label.append(i)
    test_features = test_features.drop(labels=label, axis=0)
    print(test_features.shape)

    #类型修改为object
    with open("C:/Users/ChenMingfeng/Documents/GitHub/ML-Stroke/obj_cols.txt") as f:
        c = f.read().splitlines()
        train_features[c] = train_features[c].astype("object")
        test_features[c] = test_features[c].astype("object")
    
    #缩放
    numeric_features = train_features.dtypes[train_features.dtypes == 'float64'].index
    train_features[numeric_features] = train_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    test_features[numeric_features] = test_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    
    numeric_features = train_features.dtypes[train_features.dtypes == 'int64'].index
    train_features[numeric_features] = train_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    test_features[numeric_features] = test_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    
    train_features.to_csv("heart_train_features.csv")
    test_features.to_csv("heart_test_features.csv")