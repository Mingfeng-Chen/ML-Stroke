# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:10:13 2022

@author: ChenMingfeng
"""

import pandas as pd

if __name__ == "__main__":
    X = pd.read_csv(
        "C:/Users/ChenMingfeng/Downloads/documents-export-2022-11-27/1.Stroke prediction/Stroke_prediction_system_Data/Prf_feature_train.csv").iloc[:, 1:]
    y = pd.read_csv(
        "C:/Users/ChenMingfeng/Downloads/documents-export-2022-11-27/1.Stroke prediction/Stroke_prediction_system_Data/Stroke_label_train.csv").iloc[:,1:]
    test_features = pd.read_csv(
        "C:/Users/ChenMingfeng/Downloads/documents-export-2022-11-27/1.Stroke prediction/Stroke_prediction_system_Data/Prf_feature_test.csv").iloc[:, 1:]
    print(X.shape)
    print(y.shape)
    print(test_features.shape)
    
    X = X.fillna(method='ffill')
    
    