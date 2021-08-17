# -*- coding: utf-8 -*-
import numpy as np
import catboost as cb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from get_word_vector import get_vectors
from lightgbm_classifier import get_lgb

def get_cb(train, test, use_dense=True, lgb=None, rs=None):
	# LightGBMで学習
	if not lgb:
		lgb = get_lgb(train, rs)
	# 重要度でソート
	fi = lgb.feature_importance(importance_type='split')
	inds = np.argsort(fi)[::-1]
	# 上位15個の単語を表示
	with open('voc.txt', 'r') as f:
		vocs = f.readlines()
	for i in range(15):
		print(vocs[fi[inds[i]]].strip())

	if use_dense:
		# 重要度で上位500個の単語ベクトルを作成
		imp_train = train[0][:,fi[inds[0:500]]].toarray()
		imp_test = test[0][:,fi[inds[0:500]]].toarray()
		
		# 5個の異なるアルゴリズムで100次元に次元削減したデータ5個
		pca = PCA(n_components=100, random_state=rs)
		pca_train = pca.fit_transform(train[0].toarray())
		pca_test = pca.transform(test[0].toarray())
		# tsvd = TruncatedSVD(n_components=100, random_state=rs)
		# tsvd_train = tsvd.fit_transform(train[0])
		# tsvd_test = tsvd.transform(test[0])
		# ica = FastICA(n_components=100, random_state=rs)
		# ica_train = ica.fit_transform(train[0].toarray())
		# ica_test = ica.transform(test[0].toarray())
		grp = GaussianRandomProjection(n_components=100, eps=0.1, random_state=rs)
		grp_train = grp.fit_transform(train[0])
		grp_test = grp.transform(test[0])
		srp = SparseRandomProjection(n_components=100, dense_output=True, random_state=rs)
		srp_train = srp.fit_transform(train[0])
		srp_test = srp.transform(test[0])
		
		# 合計1000次元のデータにする
		vecs_train = np.hstack([imp_train, pca_train,  grp_train, srp_train])
		vecs_test = np.hstack([imp_test, pca_test,  grp_test, srp_test])
	else:
		vecs_train = train[0]
		vecs_test = test[0]
	
	# モデルを学習
	clazz_train = train[1]
	X_train, X_test, Y_train, Y_test = train_test_split(vecs_train, clazz_train, test_size=0.1, random_state=rs)
	cb_clf = cb.train(cb.Pool(X_train, label=Y_train), 
		eval_set=cb.Pool(X_test, label=Y_test), 
		params={'loss_function':'MultiClass',
				'eval_metric':'MultiClass',
				'iterations':10,
				'learning_rate':0.1,
				'classes_count':3,
				'depth':4,
				'random_seed':rs})

	return cb_clf, vecs_test

if __name__ == '__main__':
	train, test = get_vectors()
	clf, vecs_test = get_cb(train, test, rs=1)
	
	# クラス分類を行う
	vecs, clazz = test
	clz = np.argmax(clf.predict(vecs_test), axis=1)
	report = classification_report(clazz, clz, target_names=['1次情報', '2次情報', '1.5次情報'])
	print(report)