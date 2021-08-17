# -*- coding: utf-8 -*-
import numpy as np
# from optuna.integration
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from get_word_vector import get_vectors

def get_lgb(train, rs=None):
	vecs, clazz = train
	# モデルを学習
	X_train, X_test, Y_train, Y_test = train_test_split(vecs, clazz, test_size=0.1, random_state=rs)
	lgbm_params =  {
		'task': 'train',
		'boosting_type': 'gbdt',
		'objective': 'multiclass',
		'metric': 'multi_logloss',
		'num_class': 3,
		'learning_rate': 0.05,
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
		#'num_iterations': 1000,
	}
	lgtrain = lgb.Dataset(X_train, Y_train)
	lgvalid = lgb.Dataset(X_test, Y_test)
	lgb_clf = lgb.train(
		lgbm_params,
		lgtrain,
		#num_boost_round=500,
		valid_sets=[lgtrain, lgvalid],
		#valid_names=['train','valid'],
		early_stopping_rounds=5,
		#verbose_eval=5
	)

	print(lgb_clf.params)
	print(lgb_clf.best_iteration)
	print(lgb_clf.best_score)

	return lgb_clf

if __name__ == '__main__':
	train, test = get_vectors()
	clf = get_lgb(train, rs=1)
	
	# クラス分類を行う
	vecs, clazz = test
	clz = np.argmax(clf.predict(vecs), axis=1)
	report = classification_report(clazz, clz, target_names=['1次情報', '2次情報', '1.5次情報'])
	print(report)