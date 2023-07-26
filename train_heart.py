import xgboost as xgb
import numpy as np
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve

from xgboost import XGBClassifier, plot_importance
import pickle
from datetime import datetime

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

def train_val_xgboost(train_data, train_labels, val_data, val_labels, params, iter):
    # train_data = xgb.DMatrix(train_data, train_labels)
    # val_data = xgb.DMatrix(val_data, val_labels)
    # val_data = xgb.DMatrix(val_data)

    # bst = xgb.train(params, train_data, iter)
    # bst.save_model('train_val_%s.model'%(datetime.now()))

    model = XGBClassifier(**params)
    model.fit(train_data, train_labels)

    pred = model.predict(val_data)
    print(pred)
    # import pdb;pdb.set_trace()
    # out = pred > 0.5
    # out = out + 0
    # import pdb;pdb.set_trace()
    # res = out == val_labels
    # acc = res.sum() / out.shape[0]
    recall = recall_score(pred, val_labels)
    print('Recall:', recall)

    precision = precision_score(pred, val_labels)
    print('Precision:', precision)

    acc = accuracy_score(pred, val_labels)
    print('Accuracy:', acc)

    f1 = f1_score(pred, val_labels)
    print('F1:', f1)

    fpr, tpr, thresholds = roc_curve(pred, val_labels, pos_label=1)
    AUC = auc(fpr, tpr)
    print('AUC:', AUC)

    return [recall, precision, acc, f1, AUC], model


def cross_validate_data(data, labels, k, params):
    assert len(labels.shape) == 1
    assert data.shape[0] == labels.shape[0]

    # np.random.shuffle(data)
    permt = np.random.permutation(data.shape[0])
    data = data[permt,:]
    labels = labels[permt]
    
    fold_data_num  = data.shape[0] // k
    score_list = []

    best_score = 0.0
    
    for i in range(k):
        print('Fold %d'%(i))

        start_idx = i*fold_data_num
        end_idx = (i+1)*fold_data_num
        
        val_data = data[start_idx:end_idx,:]
        val_labels = labels[start_idx:end_idx]
        print('val_data shape:', val_data.shape)
        
        train_data=np.concatenate([data[:start_idx,:],data[end_idx:,:]])
        train_labels=np.concatenate([labels[:start_idx],labels[end_idx:]])
        print('train_data shape:', train_data.shape)
        print('train_labels shape:', train_labels.shape)
        # import pdb;pdb.set_trace()
        
        score, model = train_val_xgboost(train_data=train_data, train_labels=train_labels, val_data=val_data, val_labels=val_labels, params=params, iter=100)

        if score[0] > best_score:
            best_score = score[0]
            # save
            pickle.dump(model, open("heart_cross_val_best.pkl", "wb"))
        
        score_list.append(score)
        # import pdb;pdb.set_trace()
    return np.mean(score_list, axis=0)



if __name__ == '__main__':
    # 读取csv文件
    path = "/home/rcw/ML_project/Stroke_prediction_system_Data/"
    train_feature_path = path + "Prf_feature_train.csv"
    train_label_path = path + "Stroke_label_train.csv"
    test_feature_path = path + "Prf_feature_test.csv"

    train_features = pd.read_csv('/home/rcw/ML_project/heart_train_features.csv').iloc[:, 1:]
    train_labels = pd.read_csv('/home/rcw/ML_project/heart_train_labels.csv').iloc[:, 1:]

    test_features = pd.read_csv('/home/rcw/ML_project/heart_test_features.csv').iloc[:, 1:]

    train_array = np.array(train_features)
    train_labels_array = np.array(train_labels)

    np.random.seed(2023)

    sampled_train_array, sampled_train_labels = rebalance(train_array, train_labels_array[:,0], 1, 2,2.2)
    # sampled_train_labels -= 1 # 1,2 -> 0,1
    sampled_train_labels = 2 - sampled_train_labels

    # start training scripts
    # train_data = xgb.DMatrix(sampled_train_array, label=sampled_train_labels) # 组成训练集
    # train_data.save_binary("train.buffer")

    # 划分 训练集+验证集
    X_train,X_val,y_train,y_val = train_test_split(sampled_train_array, sampled_train_labels,test_size=0.3,random_state=0)

    train_data = xgb.DMatrix(X_train, y_train)
    val_data = xgb.DMatrix(X_val)

    test_data = xgb.DMatrix(test_features) # 组成测试集
    # test_data.save_binary("heart_test.buffer")

    params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 二分类的问题
    # 'num_class': 2,               # 类别数，
    'n_estimators': 200,
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2
    'max_depth': 20,               # 构建树的深度，越大越容易过拟合
    'lambda': 4,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'min_child_weight': 6,
    'verbosity': 1,                   
    'eta': 0.05,                  # 如同学习率
    'seed': 1000,
    'subsample': 0.85,
    'colsample_bytree':0.85
    # 'nthread': 32,                  # cpu 线程数
    # 'scale_pos_weight': 0.5
    }

    num_round = 200
    # watchlist  = [(test_data,'test'),(train_data,'train')]
    # watchlist  = [(sampled_train_array,'train')]
    all_train_data = xgb.DMatrix(sampled_train_array, sampled_train_labels)

    bst = xgb.train(params, all_train_data, num_round)
    # import pdb;pdb.set_trace()
    bst.save_model('heart_train_best_%s.model'%(datetime.now()))

    pred = bst.predict(val_data)
    print(pred)
    # # import pdb;pdb.set_trace()
    out = pred > 0.5
    out = out + 0

    print('Recall:', recall_score(out, y_val))

    # res = out == y_val
    # acc = res.sum() / out.shape[0]
    # print(':', acc)
    
    # best parameters
    cross_params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 二分类的问题
    # 'num_class': 2,
    'n_estimators': 300,
    'gamma': 0.1,
    'max_depth': 20,
    'lambda': 3,
    'min_child_weight': 4,
    'verbosity': 1,                   
    'eta': 0.05,
    'seed': 1000,
    'subsample': 0.85,
    'colsample_bytree':0.85
    }

    # cross_params = {
    # 'booster': 'gbtree',
    # 'objective': 'binary:logistic',  # 二分类的问题
    # # 'num_class': 2,
    # 'n_estimators': 100,
    # 'gamma': 0.1,
    # 'max_depth': 17,
    # 'lambda': 4,
    # 'min_child_weight': 3,
    # 'verbosity': 1,                   
    # 'eta': 0.1,
    # 'seed': 1000,
    # 'subsample': 0.85,
    # 'colsample_bytree':0.85
    # # 'nthread': 32,
    # # 'scale_pos_weight': 0.5
    # }


    mean_recall, mean_precision, mean_acc, mean_f1, mean_AUC = cross_validate_data(sampled_train_array, sampled_train_labels, k=10, params=cross_params)
    
    print('mean recall:', mean_recall)
    print('mean precision:', mean_precision)
    print('mean acc:', mean_acc)
    print('mean f1:', mean_f1)
    print('mean AUC:', mean_AUC)
    


    # load
    xgb_model_loaded = pickle.load(open("heart_cross_val_best.pkl", "rb"))
    
    # import pdb;pdb.set_trace()
    # test
    print(xgb_model_loaded.predict(test_features))
