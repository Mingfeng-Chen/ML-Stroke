# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:10:13 2022

@author: ChenMingfeng & RenChengwei
"""

import pandas as pd
from sklearn.decomposition import PCA

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
    train_features.fillna(0, inplace=True)
    test_features.fillna(0, inplace=True)
    
    #类型修改为object
    with open("C:/Users/ChenMingfeng/Documents/GitHub/ML-Stroke/obj_cols.txt") as f:
        c = f.read().splitlines()
        train_features[c] = train_features[c].astype("object")
        test_features[c] = test_features[c].astype("object")
        
    #删除时间类型的特征
    temporal_cols = ['FMONTH','IDATE','IMONTH','IDAY','IYEAR','FLSHTMY3','HIVTSTD3']
    for c in temporal_cols:
        del train_features[c]
        del test_features[c]
    
    #缩放
    numeric_features = train_features.dtypes[train_features.dtypes == 'float64'].index
    train_features[numeric_features] = train_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    test_features[numeric_features] = test_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    
    numeric_features = train_features.dtypes[train_features.dtypes == 'int64'].index
    train_features[numeric_features] = train_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    test_features[numeric_features] = test_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
        
    #one-hot编码
    train_features = pd.get_dummies(train_features, dummy_na=True)
    test_features = pd.get_dummies(test_features, dummy_na=True)
    print(train_features.shape)
    print(test_features.shape)
    
    #pca降维
    pca = PCA(n_components=100)
    train_features.fillna(0, inplace=True)
    test_features.fillna(0, inplace=True)
    train_features = pd.DataFrame(pca.fit_transform(train_features))
    test_features = pd.DataFrame(pca.fit_transform(test_features))
    print(train_features.shape)
    print(test_features.shape)
    
    #写入csv文件，方便读取
    train_features.to_csv("train_features.csv")
    test_features.to_csv("test_features.csv")
    
        