# coding:utf-8
import os
from random import random
import shutil
import argparse
import sys
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import all_estimators
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-a', '--annotate')
parser.add_argument('-ho', '--holdout', action='store_true')
parser.add_argument('-k', '--kfold', action='store_true')
parser.add_argument('-c', '--comparison', action='store_true')
parser.add_argument('-nt', '--noTokenize', action='store_true')
parser.add_argument('-nv', '--noValidate', action='store_true')
parser.add_argument('-o', '--output', action='store_true')
parser.add_argument('-l', '--lemmatize', action='store_true')
parser.add_argument('-t', '--tag', action='store_true')
parser.add_argument('-p', '--pos', action='store_true')
parser.add_argument('-tp', '--tokenpos', action='store_true')
parser.add_argument('-lp', '--lemmapos', action='store_true')
parser.add_argument('-d', '--dependency', action='store_true')
parser.add_argument('-ner', '--NER', action='store_true')
args = parser.parse_args()


nlp = spacy.load('ja_ginza')
vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
target_names = ['NOT','FOUND']

if 'xlsx' in args.filename:
    df = pd.read_excel(args.filename)
elif 'csv' in args.filename:
    df = pd.read_csv(args.filename)
elif 'tsv' in args.filename:
    df = pd.read_csv(args.filename, sep='\t')

    
OUTPUT_DIR = 'result/'        


def tokenize(x_train):
    result = []
    
    for sentence in x_train:
        doc = nlp.tokenizer(str(sentence))
        sentence_tokenize = ''
        
        for token in doc:
            if args.lemmatize:
                sentence_tokenize += token.lemma_+'\n'
            elif args.tag:
                sentence_tokenize += token.tag_+'\n'
            elif args.pos:
                sentence_tokenize += token.pos_+'\n'
            elif args.tokenpos:
                sentence_tokenize += token.pos_+' '+token.orth_+'|'   
            elif args.lemmapos:
                sentence_tokenize += token.pos_+' '+token.lemma_+'|'
            elif args.dependency:
                sentence_tokenize += token.dep_+' '+token.orth_+'|'

            else:
                sentence_tokenize += token.orth_+'\n'
            
        result.append(sentence_tokenize)
        
    return result


def change_NER(x_train):

    result = []

    for sentence in x_train:
        text = list(sentence)
        doc = nlp(sentence)
        sentence_NER = ''

        entity = [ent.label_ for ent in doc.ents]       # 固有表現のラベル
        start = [ent.start_char for ent in doc.ents]    # 何文字目から固有表現か
        end = [ent.end_char for ent in doc.ents]        # 何文字目まで固有表現か
        num = 0                                        
        nowNER = False
        
        for i in range(len(text)):                      # 1文字ずつループ
            
            if (len(start) != 0) and (i == start[num]): # 固有表現の開始位置に来たら
                sentence_NER += entity[num]             # 固有表現を追加
                if num < len(start) - 1:                # out of rangeの防止
                    num += 1
                nowNER = True

            elif nowNER == True:                        # 固有表現を認識したら
                if i < end[num-1]:                      # その文字数を消費
                    continue
                elif i == end[num-1]:
                    nowNER = False
                    sentence_NER += text[i]

            else:
                sentence_NER += text[i]
        
        result.append(sentence_NER)


    return result


def create_train_data():

    x_train = df.sentence.values.astype('U').tolist()
    
    if args.NER:
        x_train = change_NER(x_train)

    if not args.noTokenize:
        x_train = tokenize(x_train)
        
    x_train = vectorizer.fit_transform(x_train)
    y_train = df.label.values.tolist()

    return x_train, y_train


def create_train_test_data(num:int=1):
    if args.annotate:
        train_df = df
        
        if 'xlsx' in args.annotate:
            test_df = pd.read_excel(args.annotate)
        elif 'csv' in args.annotate:
            test_df = pd.read_csv(args.annotate)
        elif 'tsv' in args.annotate:
            test_df = pd.read_csv(args.annotate, sep='\t')
            
        n3 = test_df.n3.values.tolist()
        vpos = test_df.vpos.values.tolist()
        easy = test_df.easy.values.tolist()
        owner = test_df.owner.values.tolist()
        
    else:
        k = int(len(df) / 10)
        dfs = [df.loc[i:i+k-1, :] for i in range(0, len(df), k)]
        
        if len(dfs) == 11:
            test_df1 = dfs.pop(-num)
            test_df2 = dfs.pop(-(num+1))
            test_df = pd.concat([test_df1, test_df2])
        elif len(dfs) == 10:
            if num != 10:
                test_df1 = dfs.pop(-num)
                test_df2 = dfs.pop(-(num+1))
                test_df = pd.concat([test_df1, test_df2])

        train_df = pd.concat(dfs)
        # train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['label'])
        train_df = train_df.replace({'NOT':0, 'FOUND':1})
        test_df = test_df.replace({'NOT':0, 'FOUND':1})
        # print(train_df.head())
    
    x_train = train_df.sentence.values.astype('U').tolist()
    x_test_raw = test_df.sentence.values.astype('U').tolist()
    
    if args.NER:
        x_train = change_NER(x_train)
        x_test_raw = change_NER(x_test_raw)

    if args.noTokenize:
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test_raw)
    else:
        x_train = tokenize(x_train)
        x_test = tokenize(x_test_raw)
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

    y_train = train_df.label.values.tolist()
    sheet_name = test_df.sheet_name.values.tolist()
    index = test_df.iloc[:,0].values.tolist()       # 0列目の値(index)をリスト化

    if args.annotate:
        y_test = -1
        return x_train, y_train, x_test, x_test_raw, y_test, sheet_name, index, n3, vpos, easy, owner
    
    else:
        y_test = test_df.label.values.tolist()
        return x_train, y_train, x_test, x_test_raw, y_test, sheet_name, index


def HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test, clf=BaggingClassifier()):
    
    clf = clf
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, predict)
    pr_auc = metrics.auc(recall, precision)

    if args.annotate is None and args.holdout:
        print(confusion_matrix(y_test, predict))
        print(classification_report(y_test, predict, target_names=target_names))
        print(f'PR_auc:{pr_auc:.3f}')
        print()


    df = pd.DataFrame()    
    df['index'] = index
    df['sheet_name'] = sheet_name
    df['sentence'] = x_test_raw
    if args.annotate is None:
        df['true'] = y_test
    df['predict'] = predict
    if args.annotate:
        df['n3'] = n3
        df['vpos'] = vpos
        df['easy'] = easy
        df['owner'] = owner

    if args.output:
        if args.holdout:
            df = df.query(" true == 0 and predict == 1 ")
            df.to_csv(f'{clf}_{num}.csv', index=False)
        else:
            df.to_csv(f'{OUTPUT_DIR}{clf}.csv', index=False)
            
            
def KFold_validation(x_train, y_train, clf=BaggingClassifier()):
    clf = clf
    score_funcs = ['accuracy','precision_macro','recall_macro','f1_macro',]
    scores = cross_validate(clf, x_train, y_train, cv=StratifiedKFold(n_splits=10), scoring = score_funcs)

    # with open('voc.txt', 'w') as f:
    #     f.write('\n'.join(vectorizer.vocabulary_.keys()))

    if args.kfold:
        print(f'classifier  :{clf}')
        print('accuracy     :{:.2f}'.format(scores['test_accuracy'].mean()))
        print('precision    :{:.2f}'.format(scores['test_precision_macro'].mean()))
        print('recall       :{:.2f}'.format(scores['test_recall_macro'].mean()))
        print('f1           :{:.2f}'.format(scores['test_f1_macro'].mean()))
        
    else:
        print('10Fold_Validation_f1:{:.2f}'.format(scores['test_f1_macro'].mean()))
        return scores['test_f1_macro'].mean()


def classifier_comparison(x_train, y_train, x_test, y_test):
        
    if args.output:
        try:
            os.makedirs(OUTPUT_DIR)
        except FileExistsError:
            # フォルダの中身を全て削除してから新規作成
            shutil.rmtree(OUTPUT_DIR)
            os.makedirs(OUTPUT_DIR)

    ranking = {}

    for name, Estimator in all_estimators(type_filter="classifier"):
        
        if name in {'CheckingClassifier', 'ClassifierChain', 'MultiOutputClassifier', 'OneVsOneClassifier', 'OneVsRestClassifier', 'OutputCodeClassifier', 'VotingClassifier', 'StackingClassifier','LogisticRegressionCV', 'RidgeClassifierCV'} :
            continue
        
        try:
            model = Estimator(class_weight='balanced')
        except:
            model = Estimator()

        print()
        print(name)
            
        try:
            model.fit(x_train,y_train)
            predict = model.predict(x_test)
        except TypeError:
            print('pass')
            continue
        except ValueError:
            print('pass')
            continue

        print()
        print(confusion_matrix(y_test, predict))
        print(classification_report(y_test, predict, digits = 2,target_names=target_names))
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, predict)
        pr_auc = metrics.auc(recall, precision)
        print(f'PR_auc:{pr_auc:.3f}')
        print()
        
        if args.output or args.annotate:
            HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test, clf=model)

        if args.noValidate:
            result = classification_report(y_test, predict, digits = 2,target_names=target_names, output_dict=True)
            
            f1 = result['macro avg']['f1-score']
            ranking[name] = round(f1, 3)
            # ranking[name] = round(pr_auc, 3)
        else:
            # f1 = KFold_validation(x_train,y_train, clf=model)
            #ranking[name] = round(f1, 3)
            ranking[name] = round(pr_auc, 3)
    
    print()
    ranking =  sorted(ranking.items(), key=lambda i: i[1], reverse=True)
    for element in ranking:
        print(element)



if __name__ == '__main__':
    
    if args.holdout:
        if args.annotate:
            x_train, y_train, x_test, x_test_raw, y_test, sheet_name, index, n3, vpos, easy, owner = create_train_test_data()
            HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test)

        else:
            if args.output:
                for num in range(1,10):
                    print(num)
                    x_train, y_train, x_test, x_test_raw, y_test, sheet_name, index = create_train_test_data(num)
                    HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test)
            else:
                x_train, y_train, x_test, x_test_raw, y_test, sheet_name, index = create_train_test_data()
                HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test)



    elif args.kfold:
        x_train, y_train = create_train_data()
        KFold_validation(x_train,y_train)
    
    
    else:
        if args.annotate:
            x_train, y_train, x_test, x_test_raw, y_test, sheet_name, index, n3, vpos, easy, owner = create_train_test_data()
        else:
            x_train, y_train, x_test, x_test_raw, y_test, sheet_name, index = create_train_test_data()
        classifier_comparison(x_train, y_train, x_test, y_test)
