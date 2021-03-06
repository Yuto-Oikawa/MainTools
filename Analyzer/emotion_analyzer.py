### GiNZA==4 ###
import re
import os
from turtle import color
import regex
import jaconv
import argparse
import collections
from pprint import pprint
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from mlask import MLAsk

import MeCab
import spacy
from spacy.matcher import Matcher
nlp = spacy.load('ja_ginza')


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--time', action='store_true')
parser.add_argument('-se', '--sentence_analyze', action='store_true')
parser.add_argument('-w', '--word_analyze')
parser.add_argument('-f', '--file_analyze')
parser.add_argument('-ba', '--bar')
parser.add_argument('-la', '--lineA', action='store_true')
parser.add_argument('-lb', '--lineB', action='store_true')
parser.add_argument('-lc', '--lineC', action='store_true')
parser.add_argument('-bo', '--both', action='store_true')
parser.add_argument('-co', '--content', action='store_true')
parser.add_argument('-d', '--dataSize', action='store_true')
parser.add_argument('-cr', '--cramer', action='store_true')
parser.add_argument('-sp', '--spearman', action='store_true')
parser.add_argument('-o', '--output', action='store_true')
args = parser.parse_args()



def get_file_name(dir:str):
    files = [f for f in os.listdir(dir)]
    files = [f for f in files if os.path.isfile(os.path.join(dir, f))]

    try:
        files.remove('.DS_Store')
    except:
        pass

    return files


def attach_time(txt_path:str):
    text = open(txt_path, errors='ignore').readlines()

    time = []
    for line in text:
        p = re.compile(r'\d{19}')
        result = p.search(line)
        
        # TweetIDがマッチしたら
        if result is not None:
            # 年月日と時刻を格納
            time.append(line[-20:-1])
    time.append(' ')

    csv_path = str(txt_path).replace('txt','csv')
    # 読み込んだcsvファイルに時刻を表す列を追加
    df2 = pd.read_csv(csv_path)
    df2['time'] = time
    df2.to_csv(csv_path, index=False)
    
    
def cramersV(x, y):
    table = np.array(pd.crosstab(x, y)).astype(np.float32)
    n = table.sum()
    colsum = table.sum(axis=0)
    rowsum = table.sum(axis=1)
    expect = np.outer(rowsum, colsum) / n
    chisq = np.sum((table - expect) ** 2 / expect)
    
    return np.sqrt(chisq / (n * (np.min(table.shape) - 1)))


def emotion_analyzer(df, filter_type:int=None, output=False, BERT=True):
    # 1つのファイルを対象に分析する関数
    df = df

    # カテゴリのフィルタリング
    label = 'pred' if BERT == True else 'label'
    try:
        if filter_type == 1:
            df = df[df[label] == 1]
        elif filter_type == 2:
            df = df[df[label] == 0]
        elif filter_type == 3:
            df = df[df[label] != 2]
        else: pass
    except KeyError:
        label = 'label'
        print('ラベルを置き換えました')
        if filter_type == 1:
            df = df[df[label] == 1]
        elif filter_type == 2:
            df = df[df[label] == 0]
        elif filter_type == 3:
            df = df[df[label] != 2]
        else: pass

    # フィルタリング後に読み込む
    text_list = df.Text.values.tolist()

    # 辞書までのパスはmecab -Dで確認可能
    emotion_analyzer = MLAsk('-d /usr/local/lib/mecab/dic/ipadic')

    # 各種のリスト
    emotion_list = []
    detail_list = []
    pojinega_list = []
    active_list = []
    hashTags = []
    cnt ={'喜':0,'好':0,'安':0,'哀':0,'嫌':0,'怒':0,'怖':0,'恥':0,'昂':0,'驚':0}
    replace ={'yorokobi':'喜','suki':'好','yasu':'安','aware':'哀','iya':'嫌','ikari':'怒','kowa':'怖','haji':'恥','takaburi':'昂','odoroki':'驚'}

    for text in text_list:
        
        # 感情分析
        try:
            analyze = emotion_analyzer.analyze(text)            
        except AttributeError:
            pass
        
        # ハッシュタグ検出
        try:
            match = re.findall(r'(#[^\s]+)', text)
            for element in match:
                hashTags.append(element)
        except TypeError:        
            pass

        # 感情語が検出された場合
        if 'representative' in analyze:    
            emotion = analyze['representative'][0]
            detail = analyze['representative'][1]
            pojinega = analyze['orientation']
            active = analyze['activation']

            #if detail != ['非難']:
            emotion_list.append(replace[emotion])
            pojinega_list.append(pojinega)
            active_list.append(active)
            detail_list.append(detail)
            cnt[replace[emotion]] += 1
            #else:
                # emotion_list.append('')
                # detail_list.append('')
                # pojinega_list.append('')
                # active_list.append('')
                
        else:
            emotion_list.append('')
            detail_list.append('')
            pojinega_list.append('')
            active_list.append('')

    
    # 1.感情カテゴリの割合をソート（args.bar）
    try:
        emo_percent = df.replace(' ', np.nan)
        #replace ={'喜':'Joy','好':'Fond','安':'Relief','哀':'Gloom','嫌':'Dislike','怒':'Anger','怖':'Fear','恥':'Shame','昂':'Excite','驚':'Surp'}
        emo_percent.replace(replace, inplace=True)
        emo_percent = emo_percent.dropna(subset=['emotion'])
        emo_percent = emo_percent['emotion'].value_counts(normalize=True) * 100
    except KeyError:
        pass
    
    # 2.頻出の感情語を表示 
    # c = collections.Counter(hashTags)
    # pprint(c.most_common(5))
    df['emotion'] = emotion_list
    df['detail'] = detail_list
    df['pojinega'] = pojinega_list
    df['activation'] = active_list
    if not (args.bar or args.cramer or args.file_analyze):
        df = df.query(' emotion=="怖" ')
        emo_count = df['detail'].value_counts()
        pprint(emo_count)
        print('全感情語',np.sum(emo_count))
        
    # 3.データ件数を表示
    dataSize = len(text_list)
    print(f'件数:{dataSize}件')
        
    # 4.クラメールの連関係数を表示
    cramer=0
    if (args.content==False) and (filter_type != 1) and (filter_type != 2) :
        try:
            cramer = cramersV(df[label], df['emotion'])
        except KeyError:
            label = 'label'
            print('ラベルを置き換えました')
            cramer = cramersV(df[label], df['emotion'])
            
        print(f'連関係数:{cramer:.3f}')
        print()
    else:
        # output==Trueの場合，df['emotion']は未定義と思われる
        print()
                
    # 5.感情語が占める割合を算出
    emotion_ratio = []
    for value in cnt.values():
        try:
            ratio = Decimal(str((value / dataSize) * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except ZeroDivisionError:
            ratio = 0
        emotion_ratio.append(ratio)

    # 6.各カテゴリの件数をリストに格納
    category_size = []
    try:
        category_size.append(len(df[df[label] == 1]))
        category_size.append(len(df[df[label] == 0]))
        category_size.append(len(df[df[label] == 2])) 
    except KeyError:
        label ='label'
        print('ラベルを置き換えました')
        category_size.append(len(df[df[label] == 1]))
        category_size.append(len(df[df[label] == 0]))
        category_size.append(len(df[df[label] == 2])) 
    
    # 7.各カテゴリの割合をリストに格納
    category_ratio = []
    for element in category_size:
        try:
            ratio = Decimal(str((element / dataSize) * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except ZeroDivisionError:
            ratio = 0
        category_ratio.append(ratio)
    
    
    # 8.必要ならcsvで出力
    if output == True:
        df.to_csv('result.csv', index=False, encoding='utf_8_sig')

    return df, emotion_ratio, category_ratio, dataSize, cramer, emo_percent


def create_ratioDF(dir, filter_type:int=None, output=False):
    # ディレクトリ内のファイル全てを対象にemotion_analyzerを適用する関数

    emotion =['喜','好','安','哀','嫌','怒','怖','恥','昂','驚']
    #emotion =['Joy', 'Fond', 'Relief', 'Gloom', 'Dislike', 'Anger', 'Fear', 'Shame', 'Excite', 'Surp']
    
    category = ['1次情報','1.5次情報','2次情報']
    df_emotion = pd.DataFrame(index=emotion)
    df_category = pd.DataFrame(index=category)

    timeNames = []
    df_list = []
    dataSize_list = []
    cramer_list = []
    files = get_file_name(dir)

    for csvName in sorted(files):
        df = pd.read_csv(dir+str(csvName))
        df2, emotion_ratio, category_ratio, dataSize, cramer, _ = emotion_analyzer(df, filter_type, output)
        df_list.append(df2)
        dataSize_list.append(dataSize)
        cramer_list.append(cramer)

        time = str(csvName).replace('.csv','') # あらかじめファイル名をツイート時刻の範囲にしておく
        timeNames.append(time.replace('2021-','').replace('-', '/'))
        df_emotion[time] = emotion_ratio                   
        df_category[time] = category_ratio
    
    if args.output:
        df = pd.concat(df_list)
        df.to_csv('result.csv', index=False, encoding='utf_8_sig')
    
    # df_emotion = df_emotion.stack()
    # df_emotion.to_csv('emotion_ratio.csv')
    # df_category = df_category.stack()
    # df_category.to_csv('category_ratio.csv')
    
    return timeNames, df_emotion, df_category, dataSize_list, cramer_list


def calc_spearman(df_emotion):
    file_num = list(range(0, len(df_emotion.loc['昂'])))
    if len(df_emotion.loc['昂']) == 1:
        print('ファイル数が1だったため，値が可視化できませんでした')
        print()

    emo_list =['昂', '怖', '安', '驚', '嫌', '好', '哀', '喜', '怒', '恥']
    #emo_list =['Excite', 'Fear', 'Relief', 'Surp', 'Dislike', 'Fond', 'Gloom', 'Joy', 'Anger', 'Shame']

    correlation_dict = {}
    
    # 全ての感情種類に対して順位相関を算出
    for emotion in emo_list:
        emotion_ratio = df_emotion.loc[emotion]
        correlation, pvalue = spearmanr(emotion_ratio, file_num)
        correlation_dict[emotion] = correlation

    # 降順にソート
    correlation_dict = dict(sorted(correlation_dict.items(), key=lambda x:x[1], reverse=True))
    correlation_list = [v for v in correlation_dict.values()]
    emo_label = [s for s in correlation_dict.keys()]
    
    return correlation_list, emo_label


def plot_data(dir:str=None, df_list:list=None, df=None, bar=False, lineA=False, lineB=False, lineC=False, content=False, dataSize=False, cramer=False, spearman=False):
    mpl.rcParams['font.family'] = 'Hiragino Maru Gothic Pro' # WindowsならYu GothicまたはMeiryo
    fig = plt.figure()

    if (dataSize or cramer) == True:
        axes_ = fig.subplots(1, 1)
    elif bar == True:
        pass     
    else:
        axes_ = fig.subplots(1, len(df_list))


    # 1 各感情の割合
    if bar:
        _, _, _, _, _, emo_percent1 = emotion_analyzer(df, filter_type=1, output=False)
        _, _, _, _, _, emo_percent2 = emotion_analyzer(df, filter_type=2, output=False)
        num = 3 if args.both else 2
        axes_ = fig.subplots(1, num)
        axes_[0].set_title('事実')
        axes_[1].set_title('その他')
        axes_[0].set_ylabel('割合[%]')

        emo_percent1[:].plot.bar(ax=axes_[0])
        emo_percent2[:].plot.bar(ax=axes_[1])
        
        axes_[0].set_yticks([0,5,10,15,20,25,30])
        axes_[1].set_yticks([0,5,10,15,20,25,30])
        
        ROTATION = 0   # Japanese
        #ROTATION = 290  # English
        SIZE = 10
        plt.setp(axes_[0].get_xticklabels(), rotation=ROTATION, fontsize=SIZE)
        plt.setp(axes_[1].get_xticklabels(), rotation=ROTATION, fontsize=SIZE)
        
        print(emo_percent1)
        print(emo_percent2)

        if args.both:
            _, _, _, _, _, emo_percent3 = emotion_analyzer(df, filter_type=3, output=False)
            axes_[2].set_title('事実+その他')
            emo_percent3[:].plot.bar(ax=axes_[2])
            plt.setp(axes_[2].get_xticklabels(), rotation=ROTATION, fontsize=SIZE)
            print(emo_percent3)


    # 2 各感情の割合の推移
    elif (lineA or lineB or lineC)==True:
        for i in range(len(df_list)):
            if lineA:
                #axes_[i].plot(timeNames,df_list[i].loc['哀'], label='哀', color='blue')
                axes_[i].plot(timeNames,df_list[i].loc['昂'], label='昂',color='gold')
                axes_[i].plot(timeNames,df_list[i].loc['驚'], label='驚',color='tomato')
                axes_[i].set_yticks([0,0.5,1.0,1.5,2.0,2.5])
                #plt.suptitle(f' 感情カテゴリの割合推移A') 
            elif lineB:
                axes_[i].plot(timeNames,df_list[i].loc['喜'], label='喜',color='orange')
                axes_[i].plot(timeNames,df_list[i].loc['安'], label='安', color='green')
                #axes_[i].plot(timeNames,df_list[i].loc['恥'], label='恥', color='magenta')
                axes_[i].set_yticks([0,1,2,3,4,5,6,7])
                #plt.suptitle(f'感情カテゴリの割合推移B')
                
            elif lineC:
                #axes_[i].plot(timeNames,df_list[i].loc['怒'], label='怒',color='red')  
                axes_[i].plot(timeNames,df_list[i].loc['嫌'], label='嫌', color='gray')
                axes_[i].plot(timeNames,df_list[i].loc['怖'], label='怖', color='black')
                axes_[i].set_yticks([4,6,8,10,12])
                #plt.suptitle(f'感情カテゴリの割合推移B')

            axes_[i].set_xticklabels(timeNames,rotation=270)
            axes_[i].legend(loc='upper right')

        axes_[0].set_ylabel('割合[%]')
        axes_[0].set_title('事実')
        axes_[1].set_title('その他')
        
        #axes_[2].set_title('1次情報 + 1.5次情報')


    # 3 内容カテゴリ
    elif content:
        if len(df_list) > 1:
            for i, dir in zip(range(len(df_list)), dir_name):
                axes_[i].plot(timeNames,df_list[i].loc['1次情報'], label='1次')
                axes_[i].plot(timeNames,df_list[i].loc['1.5次情報'], label='1.5次')
                axes_[i].plot(timeNames,df_list[i].loc['2次情報'], label='2次')
                
                axes_[i].set_title(f'{dir}')
                axes_[i].set_xticklabels(timeNames,rotation=270)
                axes_[i].legend(loc='upper right')
                axes_[0].set_ylabel('割合[%]')

        else:
            for i in range(len(df_list)):
                axes_.plot(timeNames,df_list[i].loc['1次情報'], label='1次')
                axes_.plot(timeNames,df_list[i].loc['1.5次情報'], label='1.5次')
                axes_.plot(timeNames,df_list[i].loc['2次情報'], label='2次')
                
                dir = dir.replace('/csv/', '')
                axes_.set_title(f'{dir}')
                axes_.set_xticklabels(timeNames,rotation=270)
                axes_.legend(loc='upper right')
                axes_.set_ylabel('割合[%]')

        plt.suptitle(f'内容カテゴリの割合推移 ')

    
    # 4 データ件数
    elif dataSize:
        to_english = {'#大雨':'#Downpour','#豪雨':'#Heavy rain','#大雨特別警報':'#Special storm warning','#線状降水帯':'#Linear prediction','#洪水':'#Flooding','#秋雨前線':'#Autumn Rain'}
        for dir in dir_name:
            dir += '/csv/'
            files = get_file_name(dir)
            
            files = [s.replace('.csv','').replace(':','/') for s in files]
            label = str(dir).replace('/csv/', '')
            label = to_english[label]
            dateName = sorted(files)        # 一日のツイートが1ファイルに格納されていると仮定
            dateName = [s.replace('2021-', '').replace('-','/') for s in dateName]
        
            _, _, _, dataSize_list, _ = create_ratioDF(dir, output=args.output)
            if len(dataSize_list) != 1:
                axes_.plot(dateName, dataSize_list, label=label)
            else:
                print('ファイル数が1の場合，推移が可視化できません')
                print()
        
        #axes_.set_title(f'データ件数の推移')
        plt.xticks(rotation=270)
        plt.ylabel('Number of Tweets')
        plt.legend(loc='upper right')


    # 5 クラメールの連関係数
    elif cramer:        
        cramer_dict = {}
        to_english = {'#大雨':'#Downpour','#豪雨':'#Heavy rain','#大雨特別警報':'#Special storm warning','#線状降水帯':'#Linear prediction','#洪水':'#Flooding','#秋雨前線':'#Autumn Rain'}
        for dir in dir_name:
            dir += '/csv/all/'
            _, _, _, _, cramer_list = create_ratioDF(dir, filter_type=3, output=False)
            dir = dir.replace('/csv/all/','')
            dir = to_english[dir]
            cramer_dict[dir] = np.average(cramer_list)
        
        # 降順にソート
        cramer_dict = dict(sorted(cramer_dict.items(), key=lambda x:x[1], reverse=True))
        cramer_avg = [v for v in cramer_dict.values()]
        dir_label = [s for s in cramer_dict.keys()]
        print('全データ平均:', np.average(cramer_avg))

        data_num = np.array(list(range(len(cramer_avg))))
        plt.bar(data_num, cramer_avg, tick_label=dir_label, align="center")
        #plt.title("ハッシュタグごとの連関係数")
        plt.xticks(rotation=345, fontsize=8)
        #plt.ylabel("値")
        plt.grid(True)


    # 6 スピアマンの順位相関係数
    elif spearman:
        emotion_variety = np.array(list(range(10)))
        title_name = ['事実', 'その他']

        for i, title in zip(range(len(df_list)), title_name):
            correlation_list, emo_label = calc_spearman(df_list[i])
            if len(df_list) > 1:
                axes_[i].bar(emotion_variety, correlation_list, tick_label=emo_label)
                axes_[i].set_yticks([-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8])
                axes_[i].set_title(f'{title}')
                
                #dir = dir.replace('/csv/', '')
                #plt.suptitle(f'{dir}')

            else:
                axes_.bar(emotion_variety, correlation_list, tick_label=emo_label)
                axes_.set_title(f'{dir}')
                plt.suptitle(f'スピアマンの順位相関係数 (経過時間-感情)')
        
        ROTATION = 0   # Japanese
        #ROTATION = 290  # English
        SIZE = 10
        plt.setp(axes_[0].get_xticklabels(), rotation=ROTATION, fontsize=SIZE)
        plt.setp(axes_[1].get_xticklabels(), rotation=ROTATION, fontsize=SIZE)



            
    # x軸目盛りを間引く
    # xaxis_ = axes_.xaxis
    # new_xticks = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]  # 点がない場所でも良い
    # import matplotlib.ticker as ticker
    # xaxis_.set_major_locator(ticker.FixedLocator(new_xticks))

    plt.show()


def calc_frequency_words(text_list:list):
    m = MeCab.Tagger ('-Ochasen')
    text_list = tokenize(text_list)
    
    for text in text_list:
        node = m.parseToNode(text)
        words=[]
        while node:
            hinshi = node.feature.split(",")[0]
            if hinshi in ["名詞","動詞","形容詞","副詞"]:
                origin = node.feature.split(",")[6]
                words.append(origin)
            node = node.next  
    try:
        c = collections.Counter(words)
        pprint(c.most_common(20))
        print()
    except UnboundLocalError: pass


def extract_phrase(text_list:list, word:str, bigram=True):
    text_list = tokenize(text_list)
    docs = list(nlp.pipe(text_list, disable=['ner']))

    # Matcherの生成
    matcher = Matcher(nlp.vocab)

    # パターンの追加
    #pattern = [{"POS": {"IN": ["NOUN", "PROPN"]}}, {"POS": {"IN": ["ADP"]}}, {"POS": {"IN": ["ADJ"]}},]
    #pattern = [{"POS": {"IN": ["NOUN", "PROPN"]}}, {"POS": {"IN": ["ADP"]}}, {"POS": {"IN": ["VERB"]}},]
    pattern = [{},{"TEXT": {"REGEX": word}}] if bigram else [{},{},{},{},{},{},{"TEXT": {"REGEX": word}},{}]

    matcher.add("", None, pattern)
    
    # フレーズ抽出
    text_list = []
    result_list = []
    for i, doc in enumerate(docs):

        for match_id, start, end in matcher(doc): 
            string_id = nlp.vocab.strings[match_id] # マッチしたルール名
            span = doc[start:end] # マッチしたフレーズ

            if i == 0:
                text_list.append(span.text)
                result = string_id + span.text
                print(result)
                result_list.append(result)
            else:
                if span.text in text_list:
                    continue
                else:
                    result = string_id + span.text
                    print(result)
                    result_list.append(result)
                
    print()
    
    return result_list


def normalize(_in):
    _in = regex.sub(r"〓"," ",_in)
    _in = regex.sub(r"\s\s+"," ",_in)
    _in = regex.sub(r"^\s|\s$","",_in)
    _in = jaconv.h2z(_in.upper(),kana=True,ascii=True,digit=True)
    return _in


def tokenize(sentence_list):
    result = []

    wakati = MeCab.Tagger("-r /dev/null -d  /usr/local/lib/mecab/dic/ipadic/ -Owakati")
    #wakati = MeCab.Tagger("-Owakati")

    for sentence in sentence_list:
        sentence = normalize(sentence)
        sentence_tokenize = ''
        node = wakati.parseToNode(sentence)
        
        while node:
            words_features = node.feature.split(',')
            # if '非難' in words_features[6]:
            #     node = node.next

            if words_features[6] == '*':
                sentence_tokenize += node.surface + ' '
            else:
                sentence_tokenize += words_features[6] + ' '
            node = node.next
        result.append(sentence_tokenize)
        
    return result


if __name__ == '__main__':
    dir_name = ['#大雨', '#豪雨','#大雨特別警報','#線状降水帯','#洪水', '#秋雨前線']
    dir = 'ALL/csv/'
    df_list = []

    if args.time:
        # csvファイルの列に投稿時刻を付与(txtファイルから時刻を取得)
        for dir in dir_name:
            dir += '/txt/'
            files = get_file_name(dir)
            for txt in sorted(files):
                attach_time(dir + str(txt))
                
    elif args.sentence_analyze:
        # 単文の感情解析
        emotion_analyzer = MLAsk('-d /usr/local/lib/mecab/dic/ipadic/')
        
        while True:
            try:
                text = input()
                analyze = emotion_analyzer.analyze(text)
                print(analyze)
            except EOFError:
                break
            
    elif args.file_analyze:
        df = pd.read_csv(args.file_analyze)
        emotion_analyzer(df, output=args.output)

    elif args.word_analyze:
        df = pd.read_csv(args.word_analyze)
        df = df.query('pred==1')
        df = df.query(' emotion=="喜" ')
        
        # 頻出感情語を表示
        emo_count = df['detail'].value_counts()
        pprint(emo_count)
        print('全感情語',np.sum(emo_count))
        print()
        
        # フレーズ抽出
        word = "['晴れる']"
        df = df.query(' detail==@word ')
        word = word.replace("['", "").replace("']", "")
        text_list = df.Text.values.tolist()
        result = extract_phrase(text_list, word, bigram=True)
        
        # 頻出フレーズを表示
        c = collections.Counter(result)
        pprint(c.most_common(3))
        print(len(result))
        print()
                    
        # # 頻出単語を表示
        # calc_frequency_words(result)

    elif args.bar:
        df = pd.read_csv(args.bar)
        plot_data(df=df, bar=True)

    elif (args.lineA or args.lineB or args.lineC) == True:
        task = {1:'pri', 2:'ses', 3:'both'}
        # for dir in dir_name:
        #     dir += '/csv/'

        dataSize_all = []
        for filter_type in range(1,3):
            print()
            print('emotion:', task[filter_type])
            timeNames, df_emotion, _, dataSize_list, _ = create_ratioDF(dir, filter_type, output=False)
            df_list.append(df_emotion)
            dataSize_all.append(dataSize_list)
        pprint(dataSize_all)
        
        if args.lineA:
            plot_data(dir, df_list, lineA=True)
        if args.lineB:
            plot_data(dir, df_list, lineB=True)
        if args.lineC:
            plot_data(dir, df_list, lineC=True)
        if args.output:
            print()
            print('csvのアウトプットは無効化されました')
            print('(カテゴリのフィルタリングにより，データが欠落するため)')
            print('-cまたは-dを指定して再度実行して下さい')

    elif args.content:
        # for dir in dir_name:
        #     dir += '/csv/'
        timeNames, _, df_category, _, _ = create_ratioDF(dir, output=args.output)
        df_list.append(df_category)
        
        files = get_file_name(dir)
        if len(files) < 2:
            print('警告: 一つのファイルのみが対象の場合，件数の推移を表示できません．')          
        plot_data(dir, df_list, content=True)

    elif args.dataSize:
        plot_data(dataSize=True)

    elif args.cramer:
        plot_data(cramer=True)
        if args.output:
            print()
            print('csvのアウトプットは無効化されました')
            print('(カテゴリのフィルタリングにより，データが欠落するため)')
            print('-cまたは-dを指定して再度実行して下さい')

    elif args.spearman:
        # for dir in dir_name:
        #     dir += '/csv/'
        _, df_emotion, _, _, _ = create_ratioDF(dir, filter_type=1, output=False)   # 1次
        df_list.append(df_emotion)
        _, df_emotion2, _, _, _ = create_ratioDF(dir, filter_type=2, output=False)  # 1.5次
        df_list.append(df_emotion2)
        # _, df_emotion3, _, _, _ = create_ratioDF(dir, filter_type=3, output=False)  # 1次 + 1.5次
        # df_list.append(df_emotion3)

        plot_data(dir, df_list, spearman=True)
        
        if args.output:
            print()
            print('csvのアウトプットは無効化されました')
            print('(カテゴリのフィルタリングにより，データが欠落するため)')
            print('-cまたは-dを指定して再度実行して下さい')

    else:
        print()
        print('引数を指定して下さい')
        print('(python3 emotion_analyzer.py -hで引数一覧を確認できます)')
        print()
