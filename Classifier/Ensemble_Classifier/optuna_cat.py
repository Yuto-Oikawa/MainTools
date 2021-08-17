import optuna
import numpy as np
import catboost as cb
from catboost import Pool
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from get_word_vector import get_vectors
from lightgbm_classifier import get_lgb

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



    # LightGBMで学習
    train, test = get_vectors()
    lgb = get_lgb(train, 1)
	# 重要度でソート
    fi = lgb.feature_importance(importance_type='split')
    inds = np.argsort(fi)[::-1]
	# 上位15個の単語を表示
    with open('voc.txt', 'r') as f:
        vocs = f.readlines()
    for i in range(15):
    	print(vocs[fi[inds[i]]].strip())

    # 重要度で上位500個の単語ベクトルを作成
    imp_train = train[0][:,fi[inds[0:500]]].toarray()
    imp_test = test[0][:,fi[inds[0:500]]].toarray()
    
    # 5個の異なるアルゴリズムで100次元に次元削減したデータ5個
    pca = PCA(n_components=100, random_state=1)
    pca_train = pca.fit_transform(train[0].toarray())
    pca_test = pca.transform(test[0].toarray())
    # tsvd = TruncatedSVD(n_components=100, random_state=rs)
    # tsvd_train = tsvd.fit_transform(train[0])
    # tsvd_test = tsvd.transform(test[0])
    # ica = FastICA(n_components=100, random_state=rs)
    # ica_train = ica.fit_transform(train[0].toarray())
    # ica_test = ica.transform(test[0].toarray())
    grp = GaussianRandomProjection(n_components=100, eps=0.1, random_state=1)
    grp_train = grp.fit_transform(train[0])
    grp_test = grp.transform(test[0])
    srp = SparseRandomProjection(n_components=100, dense_output=True, random_state=1)
    srp_train = srp.fit_transform(train[0])
    srp_test = srp.transform(test[0])
    
    # 合計1000次元のデータにする
    vecs_train = np.hstack([imp_train, pca_train,  grp_train, srp_train])
    vecs_test = np.hstack([imp_test, pca_test,  grp_test, srp_test])
	# else:
	# 	vecs_train = train[0]
	# 	vecs_test = test[0]
	
	# モデルを学習
    clazz_train = train[1]
    X_train, X_test, Y_train, Y_test = train_test_split(vecs_train, clazz_train, test_size=0.1, random_state=1)
    train_pool =cb.Pool(X_train,label=Y_train)
    test_pool = cb.Pool(X_test, label=Y_test)

    # パラメータの指定
    params = {
        'iterations' : trial.suggest_int('iterations', 50, 300),                         
        'depth' : trial.suggest_int('depth', 4, 9),                                       
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),               
        'random_strength' :trial.suggest_int('random_strength', 0, 100),                       
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), 
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait' :trial.suggest_int('od_wait', 10, 50)
    }

    # 学習
    model = cb.CatBoostClassifier(**params)
    model.fit(train_pool)
    # 予測
    preds = model.predict(test_pool)
    pred_labels = np.rint(preds)
    # 精度の計算
    accuracy = sklearn.metrics.accuracy_score(Y_test, pred_labels)
    return 1.0 - accuracy

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)