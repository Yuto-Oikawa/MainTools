# coding:utf-8
from gettext import ngettext
import os
import shutil
import argparse
import time
import jaconv
import regex
import spacy
import MeCab
import openpyxl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import all_estimators
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import make_scorer

#from Okapi import Okapi

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-ho', '--holdout')
parser.add_argument('-k', '--kfold', action='store_true')
parser.add_argument('-sp', '--split', action='store_true')
parser.add_argument('-a', '--annotate', action='store_true')
parser.add_argument('-c', '--comparison', action='store_true')

parser.add_argument('-all', '--all', action='store_true')
parser.add_argument('-o', '--output', action='store_true')
parser.add_argument('-m', '--mecab', action='store_true') 
parser.add_argument('-pr', '--prauc', action='store_true')
parser.add_argument('-nt', '--noTokenize', action='store_true')
parser.add_argument('-nv', '--noValidate', action='store_true')

parser.add_argument('-l', '--lemmatize', action='store_true')
parser.add_argument('-t', '--tag', action='store_true')
parser.add_argument('-po', '--pos', action='store_true')
parser.add_argument('-tp', '--tokenpos', action='store_true')
parser.add_argument('-lp', '--lemmapos', action='store_true')
parser.add_argument('-d', '--dependency', action='store_true')
parser.add_argument('-n', '--NER', action='store_true')
args = parser.parse_args()

nlp = spacy.load('ja_ginza')
vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')

target_names = ['NOT','FOUND']


def get_filename(dir:str):
    files = [f for f in os.listdir(dir)]
    files = [f for f in files if os.path.isfile(os.path.join(dir, f))]

    try:
        files.remove('.DS_Store')
    except:
        pass

    return files


def df_loader(data_name, sheet_name=None):
    if 'xlsx' in data_name:
        df = pd.read_excel(data_name, sheet_name=sheet_name)
    elif 'csv' in data_name:
        df = pd.read_csv(data_name)
    elif 'tsv' in data_name:
        df = pd.read_csv(data_name, sep='\t')
    elif 'txt':
        df = pd.read_csv(data_name, names=('sentence','label'), sep='\t')
    
    if not args.split:
        df = df.sample(frac=1, random_state=123)

    return df


def normalize(_in):
    _in = regex.sub(r"〓"," ",_in)
    _in = regex.sub(r"\s\s+"," ",_in)
    _in = regex.sub(r"^\s|\s$","",_in)
    _in = jaconv.h2z(_in.upper(),kana=True,ascii=True,digit=True)
    return _in


def tokenize(x_train):
    result = []

    if args.mecab:
        wakati = MeCab.Tagger("-r /dev/null -d  /usr/local/lib/mecab/dic/ipadic/ -Owakati")
        #wakati = MeCab.Tagger("-Owakati")

        for sentence in x_train:
            sentence = normalize(sentence)
            sentence_tokenize = ''

            if args.lemmatize:
                node = wakati.parseToNode(sentence)
                while node:
                    words_features = node.feature.split(',')

                    if words_features[6] == '*':
                        sentence_tokenize += node.surface + ' '
                    else:
                        sentence_tokenize += words_features[6] + ' '

                    node = node.next

            else:
                sentence = wakati.parse(sentence).split()
                for token in sentence:
                    sentence_tokenize += token+' '

            result.append(sentence_tokenize)

    else:
        for sentence in x_train:
            sentence = normalize(sentence)
            doc = nlp.tokenizer(str(sentence))
            sentence_tokenize = ''
            
            for token in doc:
                if args.lemmatize:
                    sentence_tokenize += token.lemma_+' '
                elif args.tag:
                    sentence_tokenize += token.tag_+' '
                elif args.pos:
                    sentence_tokenize += token.pos_+' '
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


def tokenize_okapi(x_train):
    #wakati = MeCab.Tagger("-r /dev/null -d  /usr/local/lib/mecab/dic/ipadic/ -Owakati")
    wakati = MeCab.Tagger("-Owakati")
    sentence = normalize(x_train)
    sentence = wakati.parse(sentence).split()
        
    return sentence


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
    ### used by annotate_folder(), KFold_validation() and holdout_validation()
    df = df_loader(args.filename)

    if args.kfold:
        train_df = df.replace({'NOT':0, 'FOUND':1})  
    else:
        train_df = df.replace({'NOT':'x', 'FOUND':'o'})  
    x_train = train_df.sentence.values.astype('U').tolist()
    y_train = train_df.label.values.tolist()
    
    if args.NER:
        x_train = change_NER(x_train)
    if not args.noTokenize:
        x_train = tokenize(x_train)
        
    x_train = vectorizer.fit_transform(x_train)

    return x_train, y_train


def create_test_data(data_name, sheet_name=None):
    ### used by annotate_folder(dir), holdout_validation
    if args.annotate:
        test_df = df_loader(TARGET_DIR+data_name, sheet_name=sheet_name)
    elif args.holdout:
        test_df = df_loader(data_name, sheet_name=sheet_name)

    raw_sentence = test_df.n4.values.astype('U').tolist()
    y_test = -1

    if args.NER:
        x_test = change_NER(raw_sentence)
    if not args.noTokenize:
        x_test = tokenize(raw_sentence)
    x_test = vectorizer.transform(x_test)

    try:
        # Columns to be added to the output file
        n1 = test_df.n1.values.tolist()
        n2 = test_df.n2.values.tolist()
        n3 = test_df.n3.values.tolist()
    except: return x_test, y_test, raw_sentence, -1,-1,-1


    return x_test, y_test, raw_sentence, n1, n2, n3


def create_train_test_data(num:int=0):
    df = df_loader(args.filename).replace({'NOT':'x', 'FOUND':'o'})

    if args.split:
        k = int(len(df) / 10)
        dfs = [df.loc[i:i+k-1, :] for i in range(0, len(df), k)]
        
        if len(dfs) == 11:
            test_df1 = dfs.pop(num)
            test_df2 = dfs.pop(num)
            test_df = pd.concat([test_df1, test_df2])
        elif len(dfs) == 10:
            if num != 10:
                test_df1 = dfs.pop(num)
                test_df2 = dfs.pop(num)
                test_df = pd.concat([test_df1, test_df2])
        train_df = pd.concat(dfs)

    else:
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
        # print(train_df.head())

    x_train = train_df.sentence.values.astype('U').tolist()
    raw_sentence = test_df.sentence.values.astype('U').tolist()
    
    if args.NER:
        x_train = change_NER(x_train)
        raw_sentence = change_NER(raw_sentence)

    if args.noTokenize:
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(raw_sentence)
    else:
        x_train = tokenize(x_train)
        x_test = tokenize(raw_sentence)
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        # f = tokenize_okapi
        # o = Okapi(f)
        # x_train = o.fit_transform(x_train)
        # x_test = o.transform(raw_sentence)


    y_train = train_df.label.values.tolist()
    y_test = test_df.label.values.tolist()

    try:
        # Columns to be added to the output file
        n1 = test_df.n1.values.tolist()
        n2 = test_df.n2.values.tolist()
        n3 = test_df.n3.values.tolist()
    except AttributeError:
        return x_train, y_train, x_test, y_test, raw_sentence, -1, -1, -1

    return x_train, y_train, x_test, y_test, raw_sentence, n1,n2,n3


def PR_AUC(y_true, y_pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


def holdout_validation(x_train, y_train, x_test, y_test, raw_sentence, filename=None, n1=None, n2=None, n3=None, clf=CalibratedClassifierCV() ):
    from sklearn.metrics import recall_score, precision_score, f1_score
    clf = clf
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)

    if args.split:
        print(confusion_matrix(y_test, predict))
        print(classification_report(y_test, predict, target_names=target_names))
        print(f"Micro Precision:{precision_score(y_test, predict, average='micro'):.3f}")
        print(f"Micro Recall:{recall_score(y_test, predict, average='micro'):.3f}")
        print(f"Micro f1:{f1_score(y_test, predict, average='micro'):.3f}")
        if args.prauc:
            pr_auc = PR_AUC(y_test, predict)
            print(f'PR_AUC:{pr_auc:.2f}')
            print()

    df = pd.DataFrame()
    if args.annotate:
        try:
            # Columns to be added to the output file
            df['n1'] = n1
            df['n2'] = n2
            df['n3'] = n3
        except: pass
        df['ann'] = predict
        df['res'] = ''

    else:
        df['true'] = y_test
        df['predict'] = predict

    df['sentence'] = raw_sentence
    
    if args.split:
        df = df.query(" true == 'x' and predict == 'o' ")
        #df = df.query(" true == 'o' and predict == 'x' ") 
        pass
    return df


def KFold_validation(x_train, y_train, clf=CalibratedClassifierCV()):
    clf = clf
    score_funcs = { 'accuracy': 'accuracy','precision_macro': 'precision_macro','recall_macro' :'recall_macro','f1_macro': 'f1_macro'}
    if args.prauc:
        score_funcs['PR_AUC'] = make_scorer(PR_AUC)
    scores = cross_validate(clf, x_train, y_train, cv=StratifiedKFold(n_splits=10), scoring = score_funcs)

    # with open('voc.txt', 'w') as f:
    #     f.write('\n'.join(vectorizer.vocabulary_.keys()))

    if args.kfold:
        print(f'classifier  :{clf}')
        print('accuracy     :{:.3f}'.format(scores['test_accuracy'].mean()))
        print('precision    :{:.3f}'.format(scores['test_precision_macro'].mean()))
        print('recall       :{:.3f}'.format(scores['test_recall_macro'].mean()))
        print('f1           :{:.3f}'.format(scores['test_f1_macro'].mean()))
        if args.prauc:
            print('PR_AUC       :{:.3f}'.format(scores['test_PR_AUC'].mean()))
        
    else:
        if args.prauc:
            print('PR_AUC_10Fold:{:.3f}'.format(scores['test_PR_AUC'].mean()))
            print('f1_10Fold:{:.3f}'.format(scores['test_f1_macro'].mean()))
            return scores['test_PR_AUC'].mean(), scores['test_f1_macro'].mean()
        else:
            print('f1_10Fold:{:.3f}'.format(scores['test_f1_macro'].mean()))
            return -1, scores['test_f1_macro'].mean()


def split_holdout():
    df_list = []
    for num in range(0,10,2):
        print(f'split_hold_out No:{num}')
        x_train, y_train, x_test, y_test, raw_sentence, n1,n2,n3 = create_train_test_data(num)
        result = holdout_validation(x_train, y_train, x_test, y_test, raw_sentence)
        #result = holdout_validation(x_train, y_train, x_test, y_test, raw_sentence, n1,n2,n3)

        if args.output:
            result.to_csv(f'split_{num}.tsv', sep='\t', index=False, encoding='utf_8_sig')
    
        df_list.append(result)
    df_all = pd.concat(df_list)
    #df_all.to_csv('result.tsv', sep='\t', index=False, encoding='utf_8_sig')
    df_all.to_csv(f'result.csv', index=False, encoding='utf_8_sig')



def annotate_folder():
    cnt = 0
    found_ranking  = {}
    OUTPUT_FILE = 'output.xlsx'

    files = get_filename(TARGET_DIR)
    new = openpyxl.Workbook()
    new.save(OUTPUT_FILE)

    with pd.ExcelWriter(OUTPUT_FILE, mode='a') as writer:

        for i, file in enumerate(sorted(files)):
            print(i, len(files))
            NG_count = 0

            x_test, y_test, raw_sentence, n1, n2, n3 = create_test_data(data_name=file)
            df_result = holdout_validation(x_train, y_train, x_test, y_test, raw_sentence, file)
            #df_result = holdout_validation(x_train, y_train, x_test, y_test, raw_sentence, file, n1,n2,n3)

            if args.output:
                filename = str(filename).replace('.csv', '')
                df_result.to_csv(filename+'.tsv', sep='\t', index=False, encoding='utf_8_sig')
            
            try:
                NG_count =df_result['ann'].value_counts()[1]
            except :pass

            if NG_count > 80:
                found_ranking[file] = NG_count
                sheet_name = file.replace('.csv', '')
                df_result.to_excel(writer, sheet_name=sheet_name)
                cnt += 1
    
    found_ranking =  sorted(found_ranking.items(), key=lambda i: i[1], reverse=True)
    for element in found_ranking:
        print(element)

    result = openpyxl.load_workbook(filename=OUTPUT_FILE)
    result.remove(result['Sheet'])
    result.save(OUTPUT_FILE)

    print('Added',cnt)


def classifier_comparison(x_train, y_train, x_test, y_test):
        
    if args.output:
        try:
            os.makedirs(OUTPUT_DIR)
        except FileExistsError:
            # !Delete all contents of the folder and then create a new one!
            shutil.rmtree(OUTPUT_DIR)
            os.makedirs(OUTPUT_DIR)

    ranking_pr = {}
    ranking_f1 = {}
    ranking_time = {}
    
    error_classifier = ['CheckingClassifier', 'ClassifierChain', 'MultiOutputClassifier', 'OneVsOneClassifier', 'OneVsRestClassifier', 'OutputCodeClassifier', 'VotingClassifier', 'StackingClassifier']
    low_speed_classifier = ['DecisionTreeClassifier','AdaBoostClassifier','LogisticRegressionCV','GradientBoostingClassifier','BaggingClassifier','SVC','RandomForestClassifier','ExtraTreesClassifier','MLPClassifier', 'RidgeClassifierCV', 'KNeighborsClassifier']
    if not args.all:
        error_classifier += low_speed_classifier


    for name, Estimator in all_estimators(type_filter="classifier"):
        if name in error_classifier :
            continue
        
        try:
            model = Estimator(class_weight='balanced')
        except:
            model = Estimator()

        print()
        print(name)
            
        try:
            start = time.time()
            model.fit(x_train,y_train)
            predict = model.predict(x_test)
        except TypeError:
            print('TypeError')
            continue
        except ValueError:
            print('ValueError')
            continue

        print()
        print(confusion_matrix(y_test, predict))
        print(classification_report(y_test, predict, digits = 2,target_names=target_names))

        if args.output:
            result = holdout_validation(x_train, y_train, x_test, y_test, raw_sentence, clf=model)
            result.to_csv(OUTPUT_DIR+name+'.tsv', sep='\t', index=False, encoding='utf_8_sig')

        if args.noValidate:
            result = classification_report(y_test, predict, digits = 2, target_names=target_names, output_dict=True)
            f1 = result['macro avg']['f1-score']
            ranking_f1[name] = round(f1, 3)
            if args.prauc:
                pr_auc = PR_AUC(y_test, predict)
                print(f'PR_AUC:{pr_auc:.3f}')
        else:
            pr_auc, f1 = KFold_validation(x_train,y_train, clf=model)
            ranking_pr[name] = round(pr_auc, 3)
            ranking_f1[name] = round(f1, 3)
            
        elapsed_time = time.time() - start
        ranking_time[name] = round(elapsed_time,3)
        print (f"elapsed_time:{elapsed_time:.3f}[sec]")
        print()
    
    print()
    ranking_pr =  sorted(ranking_pr.items(), key=lambda i: i[1], reverse=True)
    ranking_f1 =  sorted(ranking_f1.items(), key=lambda i: i[1], reverse=True)
    ranking_time =  sorted(ranking_time.items(), key=lambda i: i[1])


    print('time')
    for element in ranking_time:
        print(element)

    if args.prauc:
        print()
        print('PR_AUC')
        for element in ranking_pr:
                print(element)
        
    print()
    print('f1')
    for element in ranking_f1:
        print(element)


def calc_frequency_words(text_list:list):
    import collections
    from pprint import pprint

    #You can use mecab-D to check the path.
    m = MeCab.Tagger("-Owakati")
    #m = MeCab.Tagger ('-r /dev/null -d  /usr/local/lib/mecab/dic/ipadic/ -Ochasen')
    
    # for text in text_list:
        # node = m.parseToNode(str(text))
        # words=[]
        # while node:
        #     hinshi = node.feature.split(",")[0]
        #     if hinshi in ["名詞","動詞","形容詞","副詞"]:
        #         origin = node.feature.split(",")[6]
        #         words.append(origin)
        #     node = node.next  

    words = []
    text_list = tokenize(text_list)
    text_list = [s.split() for s in text_list]     
    for list in text_list:
        for word in list:
            if len(word) > 2:
                words.append(word)
    c = collections.Counter(words)
    pprint(c.most_common(20))



if __name__ == '__main__':
    TARGET_DIR = 'tgtfiles/'
    OUTPUT_DIR = 'result/'

    
    if args.holdout:
            x_train, y_train = create_train_data()
            x_test, y_test, raw_sentence, n1,n2,n3 = create_test_data(args.holdout)
            #x_train, y_train, x_test, y_test, raw_sentence, n1,n2,n3 = create_train_test_data()

            result = holdout_validation(x_train, y_train, x_test, y_test, raw_sentence)
            #_ = holdout_validation(x_train, y_train, x_test, y_test, raw_sentence, n1,n2,n3)
            result.to_csv('result.tsv', sep='\t', index=False, encoding='utf_8_sig')

    elif args.kfold:
        start = time.time()
        x_train, y_train = create_train_data()
        KFold_validation(x_train,y_train)
        elapsed_time = time.time() - start
        print (f"elapsed_time:{elapsed_time:.3f}[sec]")

    elif args.split:
        split_holdout()
        #calc_frequency_words(df_all.sentence.values.tolist())

    elif args.annotate:
        x_train, y_train = create_train_data()
        annotate_folder()

    else:
        x_train, y_train, x_test, y_test, raw_sentence, n1,n2,n3 = create_train_test_data()
        classifier_comparison(x_train, y_train, x_test, y_test)
        #classifier_comparison(x_train, y_train, x_test, y_test,n1,n2,n3)
