# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from get_word_vector import get_vectors

def get_rf(train, rs=None):
	vecs, clazz = train
	# モデルを学習
	clf = RandomForestClassifier(n_estimators=10, random_state=rs)
	clf.fit(vecs, clazz)
	return clf

if __name__ == '__main__':
	train, test = get_vectors()
	clf = get_rf(train, rs=1)
	
	# クラス分類を行う
	vecs, clazz = test
	pred = clf.predict(vecs)
	report = classification_report(clazz, pred, target_names=['1次情報', '2次情報', '1.5次情報'])
	print(report)