# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:10:13 2022

@author: ChenMingfeng & RenChengwei
"""

import pandas as pd

if __name__ == "__main__":
    #读取csv文件
    path = "C:/Users/ChenMingfeng/Downloads/documents-export-2022-11-27/1.Stroke prediction/Stroke_prediction_system_Data/"
    train_feature_path = path + "Prf_feature_train.csv"
    train_label_path = path + "Stroke_label_train.csv"
    test_feature_path = path + "Prf_feature_test.csv"
    
    #删去第一列
    train_features = pd.read_csv(train_feature_path).iloc[:, 1:]
    train_labels = pd.read_csv(train_label_path).iloc[:, 1:]
    test_features = pd.read_csv(test_feature_path).iloc[:, 1:]
    print(train_features.shape)
    print(train_labels.shape)
    print(test_features.shape)
    
    #数据预处理
    train_fetures = train_features.fillna(method='ffill')
    redundant_cols = []
