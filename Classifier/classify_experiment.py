# coding:utf-8
import os
import shutil
import argparse
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils import all_estimators
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-ho', '--holdout', action='store_true')
parser.add_argument('-k', '--kfold', action='store_true')
parser.add_argument('-c', '--comparison', action='store_true')
parser.add_argument('-nt', '--noTokenize', action='store_true')
parser.add_argument('-nv', '--noValidate', action='store_true')
parser.add_argument('-no', '--noOutput', action='store_true')
parser.add_argument('-l', '--lemmatize', action='store_true')
parser.add_argument('-p', '--pos', action='store_true')
args = parser.parse_args()


nlp = spacy.load('ja_ginza')
vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
target_names = ['NOT','FOUND']

if 'xlsx' in args.filename:
    df = pd.read_excel(args.filename)
else:
    df = pd.read_csv(args.filename, sep='\t')
    
OUTPUT_DIR = 'result/'        



def tokenize(x_train):
    result = []
    
    for line in x_train:
        doc = nlp.tokenizer(line)
        
        sentence_tokenize = ''
        for token in doc:
            sentence_tokenize += token.orth_+'\n'
            
        result.append(sentence_tokenize)
        
    return result


def create_train_data():

    x_train = df.sentence.values.tolist()
    if not args.noTokenize:
        x_train = tokenize(x_train)
    x_train = vectorizer.fit_transform(x_train)
    y_train = df.label.values.tolist()

    return x_train, y_train


def create_train_test_data():
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['label'])
    
    x_train = train_df.sentence.values.tolist()
    x_test_raw = test_df.sentence.values.tolist()
    
    if not args.noTokenize:
        x_train = tokenize(x_train)
        x_train = vectorizer.fit_transform(x_train)
        x_test = tokenize(x_test_raw)
        x_test = vectorizer.transform(x_test)
    
    else:
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test_raw)


    y_train = train_df.label.values.tolist()
    y_test = test_df.label.values.tolist()

    return x_train, y_train, x_test, x_test_raw, y_test



def HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test, clf=LinearSVC()):
    
    clf = clf
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)

    df = pd.DataFrame()
    df['sentence'] = x_test_raw
    df['true'] = y_test
    df['predict'] = predict


    if args.noOutput:
        pass
    elif args.holdout:
        df.to_csv(f'result_{clf}.csv', index=False)
        print(confusion_matrix(y_test, predict))
        print(classification_report(y_test, predict, target_names=target_names))
    else:
        df.to_csv(f'{OUTPUT_DIR}{clf}.csv', index=False)
        


def KFold_validation(x_train, y_train, clf=LinearSVC()):
    clf = clf
    score_funcs = ['accuracy','precision_macro','recall_macro','f1_macro',]
    scores = cross_validate(clf, x_train, y_train, cv=StratifiedKFold(n_splits=10), scoring = score_funcs)

    # with open('voc.txt', 'w') as f:
    #     f.write('\n'.join(vectorizer.vocabulary_.keys()))

    if args.kfold or args.noValidate:
        print(f'classifier  :{clf}')
        print('accuracy     :{:.2f}'.format(scores['test_accuracy'].mean()))
        print('precision    :{:.2f}'.format(scores['test_precision_macro'].mean()))
        print('recall       :{:.2f}'.format(scores['test_recall_macro'].mean()))
        print('f1           :{:.2f}'.format(scores['test_f1_macro'].mean()))

    else:
        print('10Fold_Validation_f1                   :{:.2f}'.format(scores['test_f1_macro'].mean()))
        return scores['test_f1_macro'].mean()


def classifier_comparison(x_train, y_train, x_test, y_test):

    ranking = {}
    
    if not args.noOutput:
        try:
            os.makedirs(OUTPUT_DIR)
        except FileExistsError:
            # フォルダの中身を全て削除してから新規作成
            shutil.rmtree(OUTPUT_DIR)
            os.makedirs(OUTPUT_DIR)


    for name, Estimator in all_estimators(type_filter="classifier"):
        if name in {'CheckingClassifier', 'ClassifierChain', 'MultiOutputClassifier', 'OneVsOneClassifier', 'OneVsRestClassifier', 'OutputCodeClassifier', 'VotingClassifier', 'StackingClassifier','LogisticRegressionCV', 'RidgeClassifierCV'} :
            continue
        model = Estimator()

        try:
            print()
            print()
            print(name)
            model.fit(x_train,y_train)
            predict = model.predict(x_test)
            print()
            print(confusion_matrix(y_test, predict))
            print(classification_report(y_test, predict, digits = 2,target_names=target_names))
            
            HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test, clf=model)

            if not args.noValidate:
                f1 = KFold_validation(x_train,y_train, clf=model)
                ranking[name] = round(f1, 3)
                
            else:
                result = classification_report(y_test, predict, digits = 2,target_names=target_names, output_dict=True)
                f1 = result['macro avg']['f1-score']
                ranking[name] = round(f1, 3)
                
            
        except TypeError:
            print('pass')
            pass
        except ValueError:
            print('pass')
            pass
    
    print()
    ranking =  sorted(ranking.items(), key=lambda i: i[1], reverse=True)
    for element in ranking:
        print(element)
    


if __name__ == '__main__':
    
    if args.holdout:
        x_train, y_train, x_test, x_test_raw, y_test = create_train_test_data()
        HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test)

    elif args.kfold:
        x_train, y_train = create_train_data()
        KFold_validation(x_train,y_train)
    
    else:
        x_train, y_train, x_test, x_test_raw, y_test = create_train_test_data()
        classifier_comparison(x_train, y_train, x_test, y_test)
