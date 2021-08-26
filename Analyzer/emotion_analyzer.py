import re
import os
import sys 
import argparse
import collections
from pprint import pprint
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mlask import MLAsk

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--time', action='store_true')
parser.add_argument('-s', '--sentence')
parser.add_argument('-o', '--output', action='store_true')
parser.add_argument('-a', '--emotionA', action='store_true')
parser.add_argument('-b', '--emotionB', action='store_true')
parser.add_argument('-c', '--content', action='store_true')
parser.add_argument('-d', '--dataSize', action='store_true')
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


def emotion_analyzer(dir, csvName:str, target_type:int=None, output=False):

    # データの読み込み
    df = pd.read_csv(dir + csvName)

    # カテゴリのフィルタリング
    if target_type in (1,4):
        df = df[df['label'] == 1]
    elif target_type in (2,5):
        df = df[df['label'] == 0]
    elif target_type in (3,6):
        df = df[df['label'] != 2]
    else: pass
    
    # テキストの読み込み
    text_list = df.Text.values.tolist()
    
    # 辞書までのパスはmecab -Dで確認可能
    emotion_analyzer = MLAsk('-d /usr/local/lib/mecab/dic/ipadic')

    # 感情語のリスト
    emotion_list = []
    detail_list = []
    pojinega_list = []
    active_list = []

    cnt ={'全感情':0, '昂':0, '怖':0, '安':0, '驚':0, '嫌':0, '好':0, '哀':0, '喜':0, '怒':0, '恥':0, ' ':0}
    replace ={'takaburi':'昂', 'kowa':'怖', 'yasu':'安', 'odoroki':'驚', 'iya':'嫌', 'suki':'好','aware':'哀','yorokobi':'喜','ikari':'怒','haji':'恥','':' '}
    hashTags = []

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

            cnt['全感情'] += 1
        else:
            emotion_list.append('')
            detail_list.append('')
            pojinega_list.append('')
            active_list.append('')


    # 頻出のハッシュタグを表示 
    c = collections.Counter(hashTags)
    pprint(c.most_common(5))

    # 感情ラベルの置き換え
    for index, emotion in enumerate(emotion_list):
        emotion_list[index] = replace[emotion]
        cnt[replace[emotion]] += 1

    # データ件数の算出
    dataSize = len(text_list) - 1
    print('件数',dataSize)
    print()
    del cnt[' ']

    # 感情語が占める割合を算出
    emotion_ratio = []
    for value in cnt.values():
        ratio = Decimal(str((value / dataSize) * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        emotion_ratio.append(ratio)


    # 必要ならcsvで出力
    if output == True:
        # 感情分析の結果を列に追加
        df['emotion'] = emotion_list
        df['detail'] = detail_list
        df['pojinega'] = pojinega_list
        df['activation'] = active_list
        df.to_csv(dir+csvName, index=False) # カテゴリ除外時は上書き注意

    return emotion_ratio, dataSize


def create_ratioDF(dir, target_type:int=None, output=False):

    emotion =['全感情', '昂', '怖', '安', '驚', '嫌', '好', '哀', '喜', '怒', '恥']
    category = ['1次情報','2次情報','1.5次情報']
    df_emotion = pd.DataFrame(index=emotion)
    df_category = pd.DataFrame(index=category)

    timeNames = []
    dataSize_list = []
    files = get_dirName(dir)

    for csvName in sorted(files):
        # 1.時刻の文字列をリストに格納
        print(str(csvName))
        time = str(csvName).replace('.csv','') # あらかじめファイル名をツイート時刻の範囲にしておく
        timeNames.append(time.replace(':','/'))            # 時刻の前半だけ格納(12:00)

        # 2.各感情語の割合をDFに格納
        emotion_ratio, dataSize = emotion_analyzer(dir,str(csvName),target_type, output)
        df_emotion[time[3:]] = emotion_ratio   # 時刻の後半も含めて列名とする(12:00_24:00)(先頭3文字のファイル番号は除外)

        # 3.各カテゴリの割合をDFに格納
        df = pd.read_csv(dir+csvName)
        category_list = []
        category_list.append(len(df[df['label'] == 1]))
        category_list.append(len(df[df['label'] == 2]))
        category_list.append(len(df[df['label'] == 0])) 

        # 4.データ件数をリストに格納
        dataSize_list.append(dataSize)

        category_ratio = []
        for element in category_list:
            ratio = Decimal(str((element / dataSize) * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            category_ratio.append(ratio)
        df_category[time[3:]] = category_ratio # 時刻の後半も格納(12:00_24:00)(ファイル番号は除外)

    # df_emotion.to_csv('emotion_ratio.csv')
    # df_category.to_csv('category_ratio.csv')

    return timeNames, dataSize_list, df_emotion, df_category


def plot_data(df_list:list, typeA=False, typeB=False, content=False, dataSize=False):
    mpl.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'    # WindowsならYu GothicまたはMeiryo
    fig = plt.figure()

    if dataSize:
        axes_ = fig.subplots(1, 1)
    else:
        axes_ = fig.subplots(1, len(df_list),)
    # axes_.plot(timeNames,df_emotion.loc['全感情'], label='全感情')


    # 感情カテゴリA
    if typeA:
        for i in range(len(df_list)):
            axes_[i].plot(timeNames,df_list[i].loc['哀'], label='哀')
            axes_[i].plot(timeNames,df_list[i].loc['昂'], label='昂')
            axes_[i].plot(timeNames,df_list[i].loc['怖'], label='怖')
            axes_[i].plot(timeNames,df_list[i].loc['嫌'], label='嫌')
            axes_[i].plot(timeNames,df_list[i].loc['驚'], label='驚')
            axes_[i].set_xticklabels(timeNames,rotation=270)
            axes_[i].legend(loc='upper right')

            if i == 0:
                axes_[i].set_ylabel('割合[%]')
                axes_[i].set_title('1次情報')
            elif i == 1:
                axes_[i].set_title('1.5次情報')
            elif i == 2:
                axes_[i].set_title('1次情報 + 1.5次情報')

            plt.suptitle('感情カテゴリの変化A')
        

    # 感情カテゴリB
    elif typeB:
        for i in range(len(df_list)):
            axes_[i].plot(timeNames,df_list[i].loc['安'], label='安')
            axes_[i].plot(timeNames,df_list[i].loc['喜'], label='喜')
            axes_[i].plot(timeNames,df_list[i].loc['好'], label='好')
            axes_[i].plot(timeNames,df_list[i].loc['怒'], label='怒')
            axes_[i].plot(timeNames,df_list[i].loc['恥'], label='恥')
            axes_[i].set_xticklabels(timeNames,rotation=270)
            axes_[i].legend(loc='upper right')

            if i == 0:
                axes_[i].set_ylabel('割合[%]')
                axes_[i].set_title('1次情報')
            elif i == 1:
                axes_[i].set_title('1.5次情報')
            elif i == 2:
                axes_[i].set_title('1次情報 + 1.5次情報')

            plt.suptitle('感情カテゴリの変化B')


    # 内容カテゴリ
    elif content:
        axes_.plot(timeNames,df_category.loc['1次情報'], label='1次')
        axes_.plot(timeNames,df_category.loc['2次情報'], label='2次')
        axes_.plot(timeNames,df_category.loc['1.5次情報'], label='1.5次')
        axes_.set_title('内容カテゴリの変化')
        plt.xticks(rotation=270)
        plt.legend(loc='upper right')

    
    # データ件数
    elif dataSize:
        
        for dir in dir_name:
            dir += '/csv/'
            files = get_dirName(dir)
            files = [s.replace('.csv','') for s in files]
            files = [s.replace(':','/') for s in files]
            label = str(dir).replace('/csv/', '')
            dateName = sorted(files)
        
            _, dataSize_list, _, _ = create_ratioDF(dir)
            axes_.plot(dateName, dataSize_list, label=label)
        
        axes_.set_title('データ件数の推移B')
        plt.xticks(rotation=270)
        plt.ylabel('件数[件]')
        plt.legend(loc='upper right')

    
    # x軸目盛りを間引く
    # xaxis_ = axes_.xaxis
    # new_xticks = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]  # 点がない場所でも良い
    # import matplotlib.ticker as ticker
    # xaxis_.set_major_locator(ticker.FixedLocator(new_xticks))


    plt.show()



if __name__ == '__main__':
    
    
    if args.time:
        # csvファイルの列に投稿時刻を付与(txtファイルから時刻を取得)
        dir_name = ['#大雨', '#豪雨', '#大雨特別警報', '#線状降水帯', ]
        dir_name2 = ['#秋雨前線', '#洪水',  '#非常に激しい雨', '#猛烈な雨']
        dir_name.extend(dir_name2)

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
        # 分析対象のファイルが存在するフォルダをdirに指定 
        dir = '#大雨/csv/'                              
        target = {1:'priA', 2:'sesA', 3:'bothA', 4:'priB', 5:'sesB', 6:'bothB', 7:'category', 8:'dataSize'}
        df_emotion_list = []
        
        if args.emotionA:
            for target_type in range(1,4):
                print(target[target_type])
                timeNames, dataSize_list, df_emotion, df_category = create_ratioDF(dir, target_type, output=False)
                df_emotion_list.append(df_emotion)
            plot_data(df_emotion_list, typeA=True)
            
            if args.output:
                print()
                print('csvのアウトプットは無効化されました')
                print('(カテゴリのフィルタリングにより，データが欠落するため)')
                print('-cまたは-dを指定して再度実行して下さい')

        
        elif args.emotionB:
            for target_type in range(4,7):
                print(target[target_type])
                timeNames, dataSize_list, df_emotion, df_category = create_ratioDF(dir, target_type, output=False)
                df_emotion_list.append(df_emotion)
            plot_data(df_emotion_list, typeB=True)
            
            if args.output:
                print()
                print('csvのアウトプットは無効化されました')
                print('(カテゴリのフィルタリングにより，データが欠落するため)')
                print('-cまたは-dを指定して再度実行して下さい')
                print()


        elif args.content:
                print(target[7])
                timeNames, dataSize_list, df_emotion, df_category = create_ratioDF(dir, output=args.output)
                df_emotion_list.append(df_emotion)
                plot_data(df_emotion_list, content=True)
                
                
        elif args.dataSize:
            dir_name = ['#大雨', '#豪雨', '#大雨特別警報', '#線状降水帯', ]
            # dir_name = ['#秋雨前線', '#洪水',  '#非常に激しい雨', '#猛烈な雨']

            print(target[8])
            for dir in dir_name:
                timeNames, dataSize_list, df_emotion, df_category = create_ratioDF(dir, output=args.output)
                df_emotion_list.append(df_emotion)
            plot_data(df_emotion_list, dataSize=True)
                
                                
        else:
            # 一つのcsvファイルを分析
            if len(sys.argv) == 2:
                dir = './'          
                _, _ = emotion_analyzer(dir, str(sys.argv[1]), output=False)
            else:
                print()
                print('引数を指定して下さい')
                print('(python3 emotion_analyzer.py -hで引数一覧を確認できます)')
                print()
