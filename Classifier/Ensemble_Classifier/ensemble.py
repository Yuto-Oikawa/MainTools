# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import classification_report

from get_word_vector import get_vectors
from randomforest_classifier import get_rf
from lightgbm_classifier import get_lgb
from xgboost_classifier import get_xgb
from catboost_classifier import get_cb

# 多数決を行う関数
def get_one(bin):
	return np.argmax(np.bincount(bin))

# アンサンブル学習
def ensemble(train, test, rs=1):
	rf_clf = get_rf(train, rs=rs)
	lgb_clf = get_lgb(train, rs=rs)
	xgb_clf, xgb_test = get_xgb(train, test, rs=rs)
	cb_clf, vecs_test = get_cb(train, test, use_dense=True, lgb=lgb_clf, rs=rs)
	
	# クラス分類を行う
	vecs, clazz = test
	clz1 = rf_clf.predict(vecs)
	clz2 = np.argmax(lgb_clf.predict(vecs), axis=1)
	clz3 = xgb_clf.predict(xgb_test)
	clz4 = np.argmax(cb_clf.predict(vecs_test), axis=1)
	clz = [get_one([clz1[i],clz2[i],clz3[i],clz4[i]]) for i in range(len(clz1))]
	return clz

if __name__ == '__main__':
	train, test = get_vectors()
	clazz = test[1]
	# アンサンブル学習1回
	print('ensemble 1:')
	clz = ensemble(train, test, rs=1)
	report = classification_report(clazz, clz, target_names=['1次情報', '2次情報', '1.5次情報'])
	print(report)
	# 乱数列を変更しながらアンサンブル学習を繰り返す
	print('random ensemble:')
	clazzes = []
	random_seeds = [1,3,7,9,13,17]
	for rs in random_seeds:
		clz = ensemble(train, test, rs=rs)
		clazzes.append(clz)
	clz = [get_one([clazzes[j][i] for j in range(len(clazzes))]) for i in range(len(clazzes[0]))]
	report = classification_report(clazz, clz, target_names=['1次情報', '2次情報', '1.5次情報'])
	print(report)