# coding:utf-8
import os
import sys
args = sys.argv

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.utils import all_estimators
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split

vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
target_names = [ '1.5次情報', '1次情報', '2次情報']
df = pd.read_excel('台風豪雨福島_tokenized.xlsx')
#df = pd.read_excel(args[2])


def createTrainData_byDirectoryText():

    dir = "train/"
    groups = [f for f in os.listdir(dir)]
    train_all  = []
    length = []

    for group in groups:
        lines = open(dir+str(group)).readlines()
        data = [line for line in lines]
        train_all.extend(data)
        length.append(len(data))

    sesquiary = length[0]
    primary = length[1]
    secondary = length[2]

    x_train = vectorizer.fit_transform(train_all)
    y_train = [0]*sesquiary + [1]*primary + [2]*secondary

    return x_train, y_train


def createTrainData_byExcel():

    x_train = df.Text.values.tolist()
    x_train = vectorizer.fit_transform(x_train)
    y_train = df.label.values.tolist()
    groups = df.type.values.tolist()

    return x_train, y_train


def createTrainTestData_byExcel():
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df[['label','type']])
    
    x_train = train_df.Text.values.tolist()
    x_train = vectorizer.fit_transform(x_train)
    x_test = test_df.Text.values.tolist()

    y_train = train_df.label.values.tolist()
    y_test = test_df.label.values.tolist()

    return x_train, y_train, x_test, y_test



def HoldOut_validation(x_train, y_train, x_test, y_test):
    
    clf = LinearSVC()
    clf.fit(x_train, y_train)
    X_test = vectorizer.transform(x_test)
    predict = clf.predict(X_test)

    df = pd.DataFrame()
    df['Text'] = x_test
    df['label'] = predict
    df.to_csv('result.csv', index=False)

    print(args[1])
    print()
    print(confusion_matrix(y_test, predict))
    print(classification_report(y_test, predict, target_names=target_names))


def KFold_validation(x_train, y_train):
    score_funcs = ['accuracy','precision_macro','recall_macro','f1_macro',]
    #scores = cross_validate(LinearSVC(), x_train, y_train, cv=GroupKFold(n_splits=3), groups = groups, scoring = score_funcs)
    scores = cross_validate(LinearSVC(), x_train, y_train, cv=StratifiedKFold(n_splits=10), scoring = score_funcs)

    print()
    print('accuracy:', scores['test_accuracy'].mean())
    print('precision:', scores['test_precision_macro'].mean())
    print('recall:', scores['test_recall_macro'].mean())
    print('f1:', scores['test_f1_macro'].mean())
    print()

    # with open('voc.txt', 'w') as f:
    #     f.write('\n'.join(vectorizer.vocabulary_.keys()))



def Classifier_Comparison(x_train, y_train, x_test, y_test):
    x_test = vectorizer.transform(x_test)

    for name, Estimator in all_estimators(type_filter="classifier"):
        if name in {'CheckingClassifier', 'ClassifierChain', 'MultiOutputClassifier', 'OneVsOneClassifier', 'OneVsRestClassifier', 'OutputCodeClassifier', 'VotingClassifier', 'LogisticRegressionCV'} :
            continue
        model = Estimator()

        try:
            model.fit(x_train,y_train)
            predict = model.predict(x_test)
            print()
            print()
            print(name)
            print(confusion_matrix(y_test, predict))
            print(classification_report(y_test, predict, digits = 2,target_names=target_names))
        except:
            pass


def print_usage():
    print()
    print('usage : Command line arguments are supported as follows')
    print()
    print('1: HoldOut_validation')
    print('2: KFold_validation')
    print('3: Classifier_Comparison')
    print()   

if __name__ == '__main__':
    
    x_train, y_train, x_test, y_test = createTrainTestData_byExcel()

    try:
        if args[1] == '1':
            HoldOut_validation(x_train, y_train, x_test, y_test)

        elif args[1] =='2':
            #x_train, y_train = createTrainData_byDirectoryText()
            x_train, y_train = createTrainData_byExcel()
            KFold_validation(x_train,y_train)
        
        elif args[1] =='3':
            Classifier_Comparison(x_train, y_train, x_test, y_test)
            
        else:
            print_usage()
    
    except:
        print_usage()