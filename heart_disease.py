# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:36:25 2022

@author: ChenMingfeng
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # 读取csv文件
    path = "C:/Users/ChenMingfeng/Downloads/documents-export-2022-11-27/1.Stroke prediction/Stroke_prediction_system_Data/"
    train_feature_path = path + "Prf_feature_train.csv"
    train_label_path = path + "Stroke_label_train.csv"
    test_feature_path = path + "Prf_feature_test.csv"

    # 删去第一列
    train_features = pd.read_csv(train_feature_path).iloc[:10000, 1:]
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
        
    #选择有心脏病的样本
    heart = []
    for i in range(train_features.shape[0]):
        if(train_features.loc[i]['HeartDisease'] != 1):
            heart.append(i)
    heart_train_features = train_features.drop(labels=heart, axis=0)
    heart_train_features.reset_index()
    
    heart = []
    for i in range(test_features.shape[0]):
        if(test_features.loc[i]['HeartDisease'] != 1):
            heart.append(i)
    heart_test_features = test_features.drop(labels=heart, axis=0)
    
    #缩放
    scaler = StandardScaler()
    scaler.fit(heart_train_features)
    scaler.fit(heart_test_features)    
            
    #聚类
    train_kmeans = KMeans(n_clusters=3).fit(heart_train_features)
    #轮廓系数
    print(silhouette_score(heart_train_features, train_kmeans.labels_, sample_size=heart_train_features.shape[0], metric='euclidean'))
        
    test_kmeans = KMeans(n_clusters=3).fit(heart_test_features)
    #轮廓系数
    print(silhouette_score(heart_test_features, test_kmeans.labels_, sample_size=heart_test_features.shape[0], metric='euclidean'))
    
    #one-hot编码
    heart_train_features = pd.get_dummies(heart_train_features, dummy_na=True)
    heart_test_features = pd.get_dummies(heart_test_features, dummy_na=True)
    
    #pca降维
    pca = PCA(n_components=100)
    heart_train_features.fillna(0, inplace=True)
    heart_test_features.fillna(0, inplace=True)
    heart_train_features = pd.DataFrame(pca.fit_transform(heart_train_features))
    heart_test_features = pd.DataFrame(pca.fit_transform(heart_test_features))
    
    heart_train_features.insert(loc=100, column='subtypes', value=train_kmeans.labels_)
    heart_test_features.insert(loc=100, column='subtypes', value=test_kmeans.labels_)
    heart_train_features['subtypes'] = heart_train_features['subtypes'].astype('object')
    heart_test_features['subtypes'] = heart_test_features['subtypes'].astype('object')
    heart_train_features = pd.get_dummies(heart_train_features)
    heart_test_features = pd.get_dummies(heart_test_features)
    print(heart_train_features.shape)
    print(heart_test_features.shape)