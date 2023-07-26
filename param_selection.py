import xgboost as xgb
import numpy as np
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier, plot_importance

import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def rebalance(data, labels, less_label, more_label, resample_rate):
    assert len(labels.shape) == 1
    assert data.shape[0] == labels.shape[0]
    assert resample_rate > 1
    less_data = data[labels == less_label]
    less_num = labels[labels == less_label].sum() // less_label
    more_num = labels.shape[0] - less_num
    assert more_num > less_num
    assert less_num * resample_rate < more_num
    more_data = data[labels != less_label]

    # print(less_data.shape)
    for i in range(math.floor(resample_rate)-1):
        if i == 0:
            rebalanced_less_data = np.concatenate([less_data, less_data])
        else:
            rebalanced_less_data = np.concatenate([rebalanced_less_data, less_data])

    random_sample_rate = resample_rate - math.floor(resample_rate)
    mask = np.random.choice([True, False], size=int(less_num), p=[random_sample_rate, 1-random_sample_rate])
    # print(mask.shape)
    tmp_less_data = less_data[mask,:]
    rebalanced_less_data = np.concatenate([rebalanced_less_data, tmp_less_data])
    rebalanced_less_labels = np.zeros(rebalanced_less_data.shape[0], dtype=np.uint8) + less_label
    print('less shape:',rebalanced_less_data.shape)

    # more downsample
    sample_prob = less_num * resample_rate / more_num
    mask = np.random.choice([True, False], size=int(more_num), p=[sample_prob, 1-sample_prob])
    sampled_more_data = more_data[mask,:]
    sampled_more_labels = np.zeros(sampled_more_data.shape[0], dtype=np.uint8) + more_label
    print('more shape:', sampled_more_data.shape)

    # shuffle
    rebalanced_data = np.concatenate([sampled_more_data, rebalanced_less_data])
    rebalanced_labels = np.concatenate([sampled_more_labels, rebalanced_less_labels])
    # np.random.shuffle(rebalanced_data)
    permt = np.random.permutation(rebalanced_data.shape[0])
    rebalanced_data = rebalanced_data[permt,:]
    rebalanced_labels = rebalanced_labels[permt]
    print('all shape:',rebalanced_data.shape)
    print('all labels shape:',rebalanced_labels.shape)

    return rebalanced_data, rebalanced_labels


def model_adjust_parameters(cv_params, other_params, X_train, y_train):


    model = XGBClassifier(**other_params)

    optimized_param = RandomizedSearchCV(estimator=model, param_distributions=cv_params, scoring='recall', cv=10, refit = True, verbose=1, n_iter=50, n_jobs = -1)

    optimized_param.fit(X_train, y_train)

    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    for mean, param in zip(means, params):
        print("mean_test_score: %f,  params: %r" % (mean, param))

    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
    print('最佳模型得分:{0}'.format(optimized_param.best_score_))


    parameters_score = pd.DataFrame(params, means)
    parameters_score['means_test_score'] = parameters_score.index
    parameters_score = parameters_score.reset_index(drop=True)
    parameters_score.to_excel('parameters_score.xlsx', index=False)
    # 画图
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 1, 1)
    plt.plot(parameters_score.iloc[:, :-1], 'o-')
    plt.legend(parameters_score.columns.to_list()[:-1], loc='upper left')
    plt.title('Parameters_size', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.subplot(2, 1, 2)
    plt.plot(parameters_score.iloc[:, -1], 'r+-')
    plt.legend(parameters_score.columns.to_list()[-1:], loc='upper left')
    plt.title('Score', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.show()





if __name__ == '__main__':
    # preprocessing
    # 读取csv文件
    path = "/home/rcw/ML_project/Stroke_prediction_system_Data/"
    train_feature_path = path + "Prf_feature_train.csv"
    train_label_path = path + "Stroke_label_train.csv"
    test_feature_path = path + "Prf_feature_test.csv"

    train_features = pd.read_csv('/home/rcw/ML_project/train_features.csv').iloc[:, 1:]
    train_labels = pd.read_csv(train_label_path).iloc[:, 1:]

    test_features = pd.read_csv('/home/rcw/ML_project/test_features.csv').iloc[:, 1:]

    train_array = np.array(train_features)
    train_labels_array = np.array(train_labels)

    np.random.seed(2023)

    sampled_train_array, sampled_train_labels = rebalance(train_array, train_labels_array[:,0], 1, 2,2.2)
    # sampled_train_labels -= 1 # 1,2 -> 0,1
    sampled_train_labels = 2 - sampled_train_labels

    # xgboost参数组合
    adj_params = {
              'n_estimators': [25, 50, 75, 100],
              'max_depth': [10, 15, 17, 20],
              'min_child_weight': [2, 4, 6],
              'lambda': [2, 6, 10],
              'eta': [0.001, 0.005, 0.01, 0.05]}
              #'gamma': [0.1, 0.2]}
	
	# 其他参数设置，每次调参将确定的参数加入，不写即默认参数
    fixed_params = {
                    'objective': 'binary:logistic',
                    'gamma' : 0.1
                    # 'num_class': 2
                    }
    # 模型调参
    model_adjust_parameters(adj_params, fixed_params, sampled_train_array, sampled_train_labels)