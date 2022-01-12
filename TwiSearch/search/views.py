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
import MeCab
import spacy
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
    length = len(lst) if len(lst) != 0 else 0
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
    x_train = df.Text.values.tolist()
    x_train = vectorizer.fit_transform(x_train)
    y_train = df.label.values.tolist()

    clf = LinearSVC()
    clf.fit(x_train, y_train)

    df_result = pd.read_csv(f'data/{word}.csv')
    raw_data = df_result.text.values.astype('U').tolist()
    
    wakati = MeCab.Tagger("-Owakati")
    # nlp = spacy.load('ja_ginza')
    x_test = []

    for sentence in raw_data:
        sentence = wakati.parse(sentence).split()
        sentence_tokenize = ''
        # doc = nlp.tokenizer(str(sentence))
        # for token in doc:
        
        for token in sentence:
            sentence_tokenize += token+'\n'
            # sentence_tokenize += token.orth_+'\n'
            
        x_test.append(sentence_tokenize)

    x_test = vectorizer.transform(x_test)
    predict = clf.predict(x_test)

    df_result['label'] = predict
    df_result.to_csv('data/'+word+'.csv', index=False)

    for i in range(len(raw_data)):
        t = Tweet(text=df_result['text'][i], label=df_result['label'][i], link=df_result['link'][i])
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

