# coding:utf-8
import os
import sys
args = sys.argv

import shutil
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.utils import all_estimators
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
import spacy
nlp = spacy.load('ja_ginza')

vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
target_names = ['NOT','FOUND']
#df = pd.read_excel(args[2])
df = pd.read_csv(args[2], sep='\t')
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


def create_train_data(tokenizer=True):

    x_train = df.sentence.values.tolist()
    if tokenizer:
        x_train = tokenize(x_train)
    x_train = vectorizer.fit_transform(x_train)
    y_train = df.label.values.tolist()

    return x_train, y_train


def create_train_test_data(tokenizer=True):
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['label'])
    
    x_train = train_df.sentence.values.tolist()
    x_test_raw = test_df.sentence.values.tolist()
    
    if tokenizer:
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



def HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test, clf=LinearSVC(), output=True):
    
    clf = clf
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)

    df = pd.DataFrame()
    df['sentence'] = x_test_raw
    df['true'] = y_test
    df['predict'] = predict

    if output:
        df.to_csv(f'result_{clf}.csv', index=False)
        print(confusion_matrix(y_test, predict))
        print(classification_report(y_test, predict, target_names=target_names))
        
    else:
        df.to_csv(f'{OUTPUT_DIR}{clf}.csv', index=False)




def KFold_validation(x_train, y_train, clf=LinearSVC(), output=True):
    clf = clf
    score_funcs = ['accuracy','precision_macro','recall_macro','f1_macro',]
    scores = cross_validate(clf, x_train, y_train, cv=StratifiedKFold(n_splits=10), scoring = score_funcs)

    # with open('voc.txt', 'w') as f:
    #     f.write('\n'.join(vectorizer.vocabulary_.keys()))

    if output:
        print(f'classifier  :{clf}')
        print('accuracy     :{:.2f}'.format(scores['test_accuracy'].mean()))
        print('precision    :{:.2f}'.format(scores['test_precision_macro'].mean()))
        print('recall       :{:.2f}'.format(scores['test_recall_macro'].mean()))
        print('f1           :{:.2f}'.format(scores['test_f1_macro'].mean()))

    else:
        print('10Fold_Validation_f1                   :{:.2f}'.format(scores['test_f1_macro'].mean()))
        return scores['test_f1_macro'].mean()


def classifier_comparison(x_train, y_train, x_test, y_test, output=True, validation=True):

    ranking = {}
    
    if output:
        try:
            os.makedirs(OUTPUT_DIR)
        except FileExistsError:
            # フォルダの中身を全て削除してから新規作成
            shutil.rmtree(OUTPUT_DIR)
            os.makedirs(OUTPUT_DIR)


    for name, Estimator in all_estimators(type_filter="classifier"):
        if name in {'CheckingClassifier', 'ClassifierChain', 'MultiOutputClassifier', 'OneVsOneClassifier', 'OneVsRestClassifier', 'OutputCodeClassifier', 'VotingClassifier', 'StackingClassifier','LogisticRegressionCV'} :
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
            
            HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test, clf=model, output=False)

            if validation:
                f1 = KFold_validation(x_train,y_train, clf=model, output=False)
                ranking[name] = round(f1, 3)
                
            else:
                result = classification_report(y_test, predict, digits = 2,target_names=target_names, output_dict=True)
                f1 = result['macro avg']['f1-score']
                ranking[name] = round(f1, 3)
                
            
        except TypeError:
            pass
        except ValueError:
            pass
    
    print()
    ranking =  sorted(ranking.items(), key=lambda i: i[1], reverse=True)
    for element in ranking:
        print(element)


def print_usage():
    print()
    print('usage : Command line arguments are supported as follows')
    print()
    print('1: HoldOut_validation')
    print('2: KFold_validation')
    print('3: classifier_comparison')
    print()   
    

if __name__ == '__main__':
    
    if args[1] == '1':
        x_train, y_train, x_test, x_test_raw, y_test = create_train_test_data(tokenizer=True)
        HoldOut_validation(x_train, y_train, x_test, x_test_raw, y_test)

    elif args[1] =='2':
        x_train, y_train = create_train_data(tokenizer=True)
        KFold_validation(x_train,y_train)
    
    elif args[1] =='3':
        x_train, y_train, x_test, x_test_raw, y_test = create_train_test_data(tokenizer=True)
        classifier_comparison(x_train, y_train, x_test, y_test, output=True, validation=True)
        
    else:
        print_usage()
