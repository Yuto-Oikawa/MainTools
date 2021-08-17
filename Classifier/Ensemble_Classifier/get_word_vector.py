# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectors():
	# 学習データを読み込む
	train1 = open('台風豪雨1次train.txt').readlines()
	train2 = open('台風豪雨2次train.txt').readlines()
	train3 = open('台風豪雨1.5次train.txt').readlines()

	# 全部繋げる
	alltxt = []
	alltxt.extend([s for s in train1])
	alltxt.extend([s for s in train2])
	alltxt.extend([s for s in train3])

	# TF-IDFベクトル化
	vectorizer = TfidfVectorizer(use_idf=True, token_pattern='(?u)\\b\\w+\\b')
	x_train= vectorizer.fit_transform(alltxt)
	# クラスを作成
	y_train= [0] * len(train1) + [1] * len(train2) + [2] * len(train3)

	
	# テスト用データを読み込む
	tst1txt = open('台風豪雨1次test.txt').readlines()
	tst2txt = open('台風豪雨2次test.txt').readlines()
	tst3txt = open('台風豪雨1.5次test.txt').readlines()
	alltxt = []
	alltxt.extend([s for s in tst1txt])
	alltxt.extend([s for s in tst2txt])
	alltxt.extend([s for s in tst3txt])

	# TF-IDFベクトル化
	x_test = vectorizer.transform(alltxt)
	# クラスを作成
	y_test = [0] * len(tst1txt) + [1] * len(tst2txt) + [2] * len(tst3txt)

	# 単語を保存
	with open('voc.txt', 'w') as f:
		f.write('\n'.join(vectorizer.vocabulary_.keys()))

	return ((x_train,y_train), (x_test,y_test))

	
	
if __name__ == '__main__':
	get_vectors()