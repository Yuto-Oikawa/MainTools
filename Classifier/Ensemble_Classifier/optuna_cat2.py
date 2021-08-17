"""
Optuna example that optimizes a classifier configuration for cancer dataset using
Catboost.
In this example, we optimize the validation accuracy of cancer detection using
Catboost. We optimize both the choice of booster model and their hyperparameters.
"""

import catboost as cb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import optuna


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


    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    gbm = cb.CatBoostClassifier(**param)

    gbm.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0, early_stopping_rounds=100)

    preds = gbm.predict(x_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))