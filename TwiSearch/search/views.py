from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from django.db.models import Q
from . import forms
from .models import Tweet

import time
import datetime
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('ja_ginza')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from data import get_tweepy as GT



def index(request):
    dim = forms.ChkForm()
    get_num = forms.get_numForm()
    context = {
        'dim': dim,
        'get_num': get_num,
    }
    template = loader.get_template('search/index.html')
    return HttpResponse(template.render(context, request))

def search_button(request):
    Tweet.objects.all().delete()
    start = time.time()
    GT.get_tweets(request.POST['word'], request.POST['start_date'], request.POST['start_time'], request.POST['end_date'], request.POST['end_time'],int(request.POST['get_num'])) 
    data_classify(request.POST['word'])

    elapsed_time = time.time() - start
    print()
    print(f'実行時間:{elapsed_time/60:.1f}min')
    print(datetime.datetime.now())
    print('およそ15分後にAPIの制限が回復します')
    print()

    return redraw(request)

def redraw(request):
    lst = create_list(request)
    length = len(lst)-1 if len(lst) != 0 else 0
    template = loader.get_template('search/result.html')
    form = forms.ChkForm()
    form.fields['dim'].initial = request.POST.getlist("dim")
    context = {
        'tweet_list': lst,
        'length': length,
        'form': form,
    }
    return HttpResponse(template.render(context, request))

def data_classify(word):

    vectorizer = TfidfVectorizer()
    df = pd.read_excel('data/train/train_tokenized.xlsx')
    train_all = df.Text.values.tolist()
    x_train = vectorizer.fit_transform(train_all)
    y_train = df.label.values.tolist()

    clf = LinearSVC()
    clf.fit(x_train, y_train)

    raw_data_tmp = open('data/'+word+'.txt', 'r', encoding='utf-8').read().split('+＝＝＝＝＝＝＝＝＝＝+')
    raw_data = [element.replace('\n','') for element in raw_data_tmp]

    with open('data/'+word+'_tokenized.txt', 'w') as f:
        for line in raw_data:
            doc = nlp.tokenizer(line)
            for token in doc:
                f.write(token.orth_.strip('\n')+' ')
            f.write('\n')

    test = open('data/'+word+'_tokenized.txt', 'r').read().splitlines()
    x_test = vectorizer.transform(test)
    predict = clf.predict(x_test)

    df = pd.DataFrame()
    df['Text'] = raw_data
    df['label'] = predict
    df.to_csv('data/result.csv', index=False)

    link = open('data/'+word+'_リンク.txt', 'r').read().split('+＝＝＝＝＝＝＝＝＝＝+')
    df['link'] = link

    for i in range(len(test)):
        t = Tweet(text=df['Text'][i], label=df['label'][i], link=df['link'][i])
        t.save()


def create_list(request):

    ret = {}

    if request.POST.getlist('dim') == ['1']:
        ret = Tweet.objects.filter(label=1)
    elif request.POST.getlist('dim') == ['2']:
        ret = Tweet.objects.filter(label=2)
    elif request.POST.getlist('dim') == ['3']:
        ret = Tweet.objects.filter(label=0)
    elif request.POST.getlist('dim') == ['1', '2']:
        ret = Tweet.objects.filter(Q(label=1) | Q(label=2))
    elif request.POST.getlist('dim') == ['1', '3']:
        ret = Tweet.objects.filter(Q(label=1) | Q(label=0))
    elif request.POST.getlist('dim') == ['2', '3']:
        ret = Tweet.objects.filter(Q(label=2) | Q(label=0))
    else:
        ret = Tweet.objects.all()

    return ret

