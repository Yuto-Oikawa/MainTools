import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import optuna

#Objective関数の設定
def objective(trial):
    # 学習データを読み込む
    train1 = open('台風豪雨1次train.txt').readlines()
    train2 = open('台風豪雨2次train.txt').readlines()
    train3 = open('台風豪雨1.5次train.txt').readlines()

    # 全部繋げる
    train_all = []
    train_all.extend([s for s in train1])
    train_all.extend([s for s in train2])
    train_all.extend([s for s in train3])

    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer(use_idf=True, token_pattern='(?u)\\b\\w+\\b')
    x_train = vectorizer.fit_transform(train_all)
    # クラスを作成
    y_train = [0] * len(train1) + [1] * len(train2) + [2] * len(train3)

    # テスト用データを読み込む
    test1 = open('台風豪雨1次test.txt').readlines()
    test2 = open('台風豪雨2次test.txt').readlines()
    test3 = open('台風豪雨1.5次test.txt').readlines()
    test_all = []
    test_all.extend([s for s in test1])
    test_all.extend([s for s in test2])
    test_all.extend([s for s in test3])


    # TF-IDFベクトル化
    x_test = vectorizer.transform(test_all)
    # クラスを作成
    y_test = [0] * len(test1) + [1] * len(test2) + [2] * len(test3)

    params = {
        #'objective': 'binary:logistic',
        'metric': 'multi_logloss',
		'num_class': 3,
        'objective': 'multi:softmax',
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0)
    }

    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, pred)
    return (1-accuracy)

if __name__ == '__main__':

    study = optuna.create_study()
    study.optimize(objective, n_trials=300)

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)