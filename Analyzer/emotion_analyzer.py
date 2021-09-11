import re
import os
import sys 
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

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--time', action='store_true')
parser.add_argument('-se', '--sentence')
parser.add_argument('-o', '--output', action='store_true')
parser.add_argument('-a', '--emotionA', action='store_true')
parser.add_argument('-b', '--emotionB', action='store_true')
parser.add_argument('-co', '--content', action='store_true')
parser.add_argument('-d', '--dataSize', action='store_true')
parser.add_argument('-cr', '--cramer', action='store_true')
parser.add_argument('-sp', '--spearman', action='store_true')
parser.add_argument('-si', '--single')

args = parser.parse_args()



def get_dirName(dir:str):
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


def emotion_analyzer(dir, csvName:str, filter_type:int=None, output=False, BERT=True):
    # 1つのファイルを対象に分析する関数

    # データの読み込み
    df = pd.read_csv(dir + csvName)
    df = df.dropna(subset=['Text'])

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
    cnt ={'昂':0, '怖':0, '安':0, '驚':0, '嫌':0, '好':0, '哀':0, '喜':0, '怒':0, '恥':0, ' ':0}
    replace ={'takaburi':'昂', 'kowa':'怖', 'yasu':'安', 'odoroki':'驚', 'iya':'嫌', 'suki':'好','aware':'哀','yorokobi':'喜','ikari':'怒','haji':'恥','':' '}

    for index, text in enumerate(text_list):
        
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

            emotion_list.append(emotion)
            detail_list.append(detail)
            pojinega_list.append(pojinega)
            active_list.append(active)
            #cnt['全感情'] += 1
        else:
            emotion_list.append('')
            detail_list.append('')
            pojinega_list.append('')
            active_list.append('')


    # 1.ファイル名を表示
    print(str(csvName))
    time = str(csvName).replace('.csv','') # あらかじめファイル名をツイート時刻の範囲にしておく

    # 2.頻出のハッシュタグを表示 
    c = collections.Counter(hashTags)
    pprint(c.most_common(5))

    # 3.データ件数を表示
    dataSize = len(text_list)
    print(f'件数:{dataSize}件',)
        
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
            
    # 5.感情ラベルの置き換え
    for index, emotion in enumerate(emotion_list):
        emotion_list[index] = replace[emotion]
        cnt[replace[emotion]] += 1
    del cnt[' ']
    
    # 6.感情語が占める割合を算出
    emotion_ratio = []
    for value in cnt.values():
        try:
            ratio = Decimal(str((value / dataSize) * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except ZeroDivisionError:
            ratio = 0
        emotion_ratio.append(ratio)

    # 7.各カテゴリの件数をリストに格納
    category_size = []
    try:
        category_size.append(len(df[df[label] == 1]))
        category_size.append(len(df[df[label] == 2]))
        category_size.append(len(df[df[label] == 0])) 
    except KeyError:
        label ='label'
        print('ラベルを置き換えました')
        category_size.append(len(df[df[label] == 1]))
        category_size.append(len(df[df[label] == 2]))
        category_size.append(len(df[df[label] == 0])) 
    
    # 8.各カテゴリの割合をリストに格納
    category_ratio = []
    for element in category_size:
        try:
            ratio = Decimal(str((element / dataSize) * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except ZeroDivisionError:
            ratio = 0
        category_ratio.append(ratio)
    
    # 9. 感情カテゴリの割合を表示
    try:
        print(df['emotion'].value_counts())
    except KeyError:
        pass

    # 10.必要ならcsvで出力
    if output == True:
        # 感情分析の結果を列に追加
        df['emotion'] = emotion_list
        df['detail'] = detail_list
        df['pojinega'] = pojinega_list
        df['activation'] = active_list
        df.to_csv(dir+csvName, index=False)

    return time, emotion_ratio, category_ratio, dataSize, cramer


def create_ratioDF(dir, filter_type:int=None, output=False):
    # ディレクトリ内のファイル全てを対象にemotion_analyzerを適用する関数

    emotion =['昂', '怖', '安', '驚', '嫌', '好', '哀', '喜', '怒', '恥']
    category = ['1次情報','2次情報','1.5次情報']
    df_emotion = pd.DataFrame(index=emotion)
    df_category = pd.DataFrame(index=category)

    timeNames = []
    dataSize_list = []
    cramer_list = []
    files = get_dirName(dir)

    for csvName in sorted(files):

        time, emotion_ratio, category_ratio, dataSize, cramer = emotion_analyzer(dir,str(csvName),filter_type, output)
        timeNames.append(time.replace(':','/'))             # 時刻の前半だけ格納(12:00)
        df_emotion[time] = emotion_ratio                    # 時刻の後半も含めて列名とする(12:00_24:00)(先頭3文字のファイル番号を除外するならtime[3:])
        df_category[time] = category_ratio                  # 時刻の後半も格納(12:00_24:00)(ファイル番号は除外)
        dataSize_list.append(dataSize)
        cramer_list.append(cramer)

    # df_emotion.to_csv('emotion_ratio.csv')
    # df_category.to_csv('category_ratio.csv')
    
    return timeNames, df_emotion, df_category, dataSize_list, cramer_list


def calc_spearman(df_emotion):
    file_num = list(range(0, len(df_emotion.loc['怖'])))
    if len(df_emotion.loc['怖']) == 1:
        print('ファイル数が1だったため，値が可視化できませんでした')
        print()

    emo_list =['昂', '怖', '安', '驚', '嫌', '好', '哀', '喜', '怒', '恥']
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


def plot_data(dir:str=None, df_list:list=None, typeA=False, typeB=False, content=False, dataSize=False, cramer=False, spearman=False):
    mpl.rcParams['font.family'] = 'Hiragino Maru Gothic Pro' # WindowsならYu GothicまたはMeiryo
    fig = plt.figure()

    if (dataSize or cramer) == True:
        axes_ = fig.subplots(1, 1)
    else:
        axes_ = fig.subplots(1, len(df_list))
    # axes_.plot(timeNames,df_emotion.loc['全感情'], label='全感情')


    # 感情カテゴリ
    if (typeA or typeB)==True:
        for i in range(len(df_list)):
            if typeA:
                axes_[i].plot(timeNames,df_list[i].loc['哀'], label='哀')
                # axes_[i].plot(timeNames,df_list[i].loc['昂'], label='昂')
                # axes_[i].plot(timeNames,df_list[i].loc['怖'], label='怖')
                # axes_[i].plot(timeNames,df_list[i].loc['嫌'], label='嫌')
                # axes_[i].plot(timeNames,df_list[i].loc['驚'], label='驚')
                plt.suptitle(f' 感情カテゴリの割合推移A ({dir[:-5]})') # [:-5]は/csv/の除去
            elif typeB:
                axes_[i].plot(timeNames,df_list[i].loc['安'], label='安')
                axes_[i].plot(timeNames,df_list[i].loc['喜'], label='喜')
                axes_[i].plot(timeNames,df_list[i].loc['好'], label='好')
                axes_[i].plot(timeNames,df_list[i].loc['怒'], label='怒')
                axes_[i].plot(timeNames,df_list[i].loc['恥'], label='恥')
                plt.suptitle(f'感情カテゴリの割合推移B ({dir[:-5]})')

            axes_[i].set_xticklabels(timeNames,rotation=270)
            axes_[i].legend(loc='upper right')

        axes_[0].set_ylabel('割合[%]')
        axes_[0].set_title('1次情報')
        axes_[1].set_title('1.5次情報')
        axes_[2].set_title('1次情報 + 1.5次情報')


    # 内容カテゴリ
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

    
    # データ件数
    elif dataSize:
        
        for dir in dir_name:
            dir += '/csv/'
            files = get_dirName(dir)
            
            files = [s.replace('.csv','').replace(':','/') for s in files]
            label = str(dir).replace('/csv/', '')
            dateName = sorted(files)        # 一日のツイートが1ファイルに格納されていると仮定
        
            _, _, _, dataSize_list, _ = create_ratioDF(dir, output=args.output)
            if len(dataSize_list) != 1:
                axes_.plot(dateName, dataSize_list, label=label)
            else:
                print('ファイル数が1の場合，推移が可視化できません')
                print()
        
        #axes_.set_title(f'データ件数の推移')
        plt.xticks(rotation=270)
        plt.ylabel('件数[件]')
        plt.legend(loc='upper right')


    # クラメールの連関係数
    elif cramer:        
        cramer_dict = {}
        for dir in dir_name:
            dir += '/csv/all/'
            _, _, _, _, cramer_list = create_ratioDF(dir, filter_type=3, output=False)
            cramer_dict[dir] = np.average(cramer_list)
        
        # 降順にソート
        cramer_dict = dict(sorted(cramer_dict.items(), key=lambda x:x[1], reverse=True))
        cramer_avg = [v for v in cramer_dict.values()]
        dir_label = [s.replace('/csv/all/','') for s in cramer_dict.keys()]
        print('全データ平均:', np.average(cramer_avg))

        data_num = np.array(list(range(len(cramer_avg))))
        plt.bar(data_num, cramer_avg, tick_label=dir_label, align="center")
        #plt.title("ハッシュタグごとの連関係数")
        plt.ylabel("値")
        plt.grid(True)

    
    # スピアマンの順位相関係数
    elif spearman:
        emotion_variety = np.array(list(range(10)))
        
        for i, title in zip(range(len(df_list)), title_name):
            correlation_list, emo_label = calc_spearman(df_list[i])
            if len(df_list) > 1:
                axes_[i].bar(emotion_variety, correlation_list, tick_label=emo_label)
                axes_[i].set_title(f'{title}')
                dir = dir.replace('/csv/', '')
                #plt.suptitle(f'{dir}')

            else:
                axes_.bar(emotion_variety, correlation_list, tick_label=emo_label)
                axes_.set_title(f'{dir}')
                plt.suptitle(f'スピアマンの順位相関係数 (経過時間-感情)')

            
            

    # x軸目盛りを間引く
    # xaxis_ = axes_.xaxis
    # new_xticks = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]  # 点がない場所でも良い
    # import matplotlib.ticker as ticker
    # xaxis_.set_major_locator(ticker.FixedLocator(new_xticks))

    plt.show()
    
    

if __name__ == '__main__':
    dir_name = ['#大雨', '#豪雨','#秋雨前線' ,'#線状降水帯', '#洪水', '#大雨特別警報', 'ALL']
    dir = 'ALL/csv/'

    if args.time:
        # csvファイルの列に投稿時刻を付与(txtファイルから時刻を取得)
        for dir in dir_name:
            dir += '/txt/'
            files = get_dirName(dir)
            for txt in sorted(files):
                attach_time(dir + str(txt))
                
                
    elif args.sentence is not None:
        # 単文の感情解析
        emotion_analyzer = MLAsk('-d /usr/local/lib/mecab/dic/ipadic/')
        analyze = emotion_analyzer.analyze(args.sentence)
        print(analyze)


    else:        
        task = {1:'pri', 2:'ses', 3:'both', 4:'category', 5:'dataSize', 6:'cramer', 7:'spearman'}
        df_list = []
        
        if (args.emotionA or args.emotionB) == True:
            # for dir in dir_name:
            #     dir += '/csv/'

            dataSize_all = []
            for filter_type in range(1,4):
                print()
                print('emotion:', task[filter_type])
                timeNames, df_emotion, _, dataSize_list, _ = create_ratioDF(dir, filter_type, output=False)
                df_list.append(df_emotion)
                dataSize_all.append(dataSize_list)
            pprint(dataSize_all)
                
            if args.emotionA:
                plot_data(dir, df_list, typeA=True)
            if args.emotionB:
                plot_data(dir, df_list, typeB=True)
            if args.output:
                print()
                print('csvのアウトプットは無効化されました')
                print('(カテゴリのフィルタリングにより，データが欠落するため)')
                print('-cまたは-dを指定して再度実行して下さい')


        elif args.content:
            # for dir in dir_name:
            #     dir += '/csv/'
            print(task[4])
            timeNames, _, df_category, _, _ = create_ratioDF(dir, output=args.output)
            df_list.append(df_category)
            
            files = get_dirName(dir)
            if len(files) < 2:
                print('警告: 一つのファイルのみが対象の場合，件数の推移を表示できません．')
                    
            plot_data(dir, df_list, content=True)
                    
                
        elif args.dataSize:
            print(task[5])
            plot_data(dataSize=True)

        
        elif args.cramer:
            print(task[6])
            plot_data(cramer=True)
            if args.output:
                print()
                print('csvのアウトプットは無効化されました')
                print('(カテゴリのフィルタリングにより，データが欠落するため)')
                print('-cまたは-dを指定して再度実行して下さい')


        elif args.spearman:
            title_name = ['1次', '1.5次', '1次 + 1.5次']
            print(task[7])
            # for dir in dir_name:
            #     dir += '/csv/'
            _, df_emotion, _, _, _ = create_ratioDF(dir, filter_type=1, output=False)   # 1次
            df_list.append(df_emotion)
            _, df_emotion2, _, _, _ = create_ratioDF(dir, filter_type=2, output=False)  # 1.5次
            df_list.append(df_emotion2)
            _, df_emotion3, _, _, _ = create_ratioDF(dir, filter_type=3, output=False)  # 1次 + 1.5次
            df_list.append(df_emotion3)

                
            plot_data(dir, df_list, spearman=True)
            
            if args.output:
                print()
                print('csvのアウトプットは無効化されました')
                print('(カテゴリのフィルタリングにより，データが欠落するため)')
                print('-cまたは-dを指定して再度実行して下さい')


        elif args.single is not None:
            dir = './'          
            _, _, _, _, _ = emotion_analyzer(dir, args.single, filter_type=3, output=False)
                
        else:
            print()
            print('引数を指定して下さい')
            print('(python3 emotion_analyzer.py -hで引数一覧を確認できます)')
            print()
