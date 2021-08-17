# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from get_word_vector import get_vectors

def get_xgb(train, test, rs=None):
	vecs, clazz = train
	# モデルを学習
	X_train, X_test, Y_train, Y_test = train_test_split(vecs, clazz, test_size=0.1, random_state=rs)
	xgtrain = xgb.DMatrix(X_train, Y_train)
	xgvalid = xgb.DMatrix(X_test, Y_test)
	xgb_params = {
		'task': 'train',
		'boosting_type': 'gbdt',
		'objective': 'multi:softmax',
		'metric': 'multi_logloss',
		'num_class': 3,
		'learning_rate': 0.5775219366078943,
		'verbose': 0,
		'force_col_wise':True,
		'feature_pre_filter':False,
		'lambda_l1': 0.0006535311486910473,
		'lambda_l2': 2.3976476190395467e-05,
		'num_leaves': 31,
		'feature_fraction': 0.5,
		'bagging_fraction': 0.8960668417864669,
		'bagging_freq': 4,
		'min_child_samples': 20,
		'max_depth': 2,
		'n_estimators': 228,
	}

	xgb_clf = xgb.train(
		xgb_params,
		xgtrain,
		30, 
		[(xgtrain,'train'), (xgvalid,'valid')],
		maximize=False,
		verbose_eval=10, 
		early_stopping_rounds=10
	)

	return xgb_clf, xgb.DMatrix(test[0])

if __name__ == '__main__':
	train, test = get_vectors()
	clf, xgb_test = get_xgb(train, test, rs=1)
	
	# クラス分類を行う
	vecs, clazz = test
	clz = clf.predict(xgb_test)
	report = classification_report(clazz, clz, target_names=['1次情報', '2次情報', '1.5次情報'])
	print(report)