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
parser.add_argument('-e', '--emotion', action='store_true')
args = parser.parse_args()



def get_dirName(dir:str):
    files = [f for f in os.listdir(dir)]

    try:
        files.remove('.DS_Store')
    except:
        pass

    return files


def attach_time(txtName:str):
    read_dir = 'txt/'       # 時刻の書かれたtxtファイルが格納されているディレクトリ
    text = open(read_dir + txtName).readlines()

    time = []
    for line in text:
        p = re.compile(r'\d{19}')
        result = p.search(line)
        
        # 時間がマッチしたら
        if result is not None:
            # 時分秒のみ格納
            time.append(line[-9:-1])
    time.append(' ')

    csvName = str(txtName).replace('.txt','.csv')
    write_dir = 'csv/'     # 時刻を付与したいcsvファイルが格納されているディレクトリ

    # 読み込んだcsvファイルに時刻を表す列を追加
    df2 = pd.read_csv(write_dir + csvName)
    df2['time'] = time
    df2.to_csv(write_dir + csvName, index=False)


def emotion_analyzer(csvName:str, target_type=7, output=False):

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


def create_ratioDF(target_type:int, output=False):

    emotion =['全感情', '昂', '怖', '安', '驚', '嫌', '好', '哀', '喜', '怒', '恥']
    category = ['1次情報','2次情報','1.5次情報']
    df_emotion = pd.DataFrame(index=emotion)
    df_category = pd.DataFrame(index=category)

    timeNames = []
    files = get_dirName(dir)

    for csvName in sorted(files):
        # 1.時刻の文字列をリストに格納
        print(str(csvName))
        time = str(csvName).replace('.csv','') # あらかじめファイル名をツイート時刻の範囲にしておく
        timeNames.append(time[3:8])            # 時刻の前半だけ格納(12:00)

        # 2.各感情語の割合をDFに格納
        emotion_ratio, dataSize = emotion_analyzer(str(csvName),target_type, output)
        df_emotion[time[3:]] = emotion_ratio   # 時刻の後半も含めて列名とする(12:00_24:00)(先頭3文字のファイル番号は除外)

        # 3.各カテゴリの割合をDFに格納
        df = pd.read_csv(dir+csvName)
        category_list = []
        category_list.append(len(df[df['label'] == 1]))
        category_list.append(len(df[df['label'] == 2]))
        category_list.append(len(df[df['label'] == 0]))

        category_ratio = []
        for element in category_list:
            ratio = Decimal(str((element / dataSize) * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            category_ratio.append(ratio)
        df_category[time[3:]] = category_ratio # 時刻の後半も格納(12:00_24:00)(ファイル番号は除外)

    # df_emotion.to_csv('emotion_ratio.csv')
    # df_category.to_csv('category_ratio.csv')

    return timeNames, df_emotion, df_category


def plot_data(target_type:int):
    mpl.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'    # WindowsならYu GothicまたはMeiryo
    fig = plt.figure()

    axes_ = fig.add_subplot(1, 1, 1,)
    # axes_.plot(timeNames,df_emotion.loc['全感情'], label='全感情')

    if target_type < 4:
        axes_.plot(timeNames,df_emotion.loc['哀'], label='哀')
        axes_.plot(timeNames,df_emotion.loc['昂'], label='昂')
        axes_.plot(timeNames,df_emotion.loc['怖'], label='怖')
        axes_.plot(timeNames,df_emotion.loc['嫌'], label='嫌')
        axes_.plot(timeNames,df_emotion.loc['驚'], label='驚')

        if target_type == 1:
            axes_.set_title('感情カテゴリの変化A(1次情報)')
        elif target_type == 2:
            axes_.set_title('感情カテゴリの変化A(1.5次情報)')
        elif target_type == 3:
            axes_.set_title('感情カテゴリの変化A(1次情報+1.5次情報)')

    elif target_type < 7:
        axes_.plot(timeNames,df_emotion.loc['安'], label='安')
        axes_.plot(timeNames,df_emotion.loc['喜'], label='喜')
        axes_.plot(timeNames,df_emotion.loc['好'], label='好')
        axes_.plot(timeNames,df_emotion.loc['怒'], label='怒')
        axes_.plot(timeNames,df_emotion.loc['恥'], label='恥')

        if target_type == 4:
            axes_.set_title('感情カテゴリの変化B(1次情報)')
        elif target_type == 5:
            axes_.set_title('感情カテゴリの変化B(1.5次情報)')
        elif target_type == 6:
            axes_.set_title('感情カテゴリの変化B(1次情報+1.5次情報)')

    else:
        axes_.plot(timeNames,df_category.loc['1次情報'], label='1次')
        axes_.plot(timeNames,df_category.loc['2次情報'], label='2次')
        axes_.plot(timeNames,df_category.loc['1.5次情報'], label='1.5次')
        axes_.set_title('内容カテゴリの変化')


    # x軸目盛りを間引く
    # xaxis_ = axes_.xaxis
    # new_xticks = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]  # 点がない場所でも良い
    # import matplotlib.ticker as ticker
    # xaxis_.set_major_locator(ticker.FixedLocator(new_xticks))

    plt.xticks(rotation=270)
    plt.ylabel('割合[%]')
    plt.legend(loc='upper right')
    plt.show()



if __name__ == '__main__':
    
    if args.time:
        # csvファイルの列に投稿時刻を付与(txtファイルから時刻を取得)
        files = get_dirName('txt/')
        for txt in sorted(files):
            attach_time(str(txt))

    elif args.emotion:
        # 単文の感情解析
        emotion_analyzer = MLAsk('-d /usr/local/lib/mecab/dic/ipadic/')
        s = '怖い'
        analyze = emotion_analyzer.analyze(s)
        print(analyze)
        

    else:
        # 一つのcsvファイルを分析
        if len(sys.argv) == 2:
            dir = './'          
            _, _ = emotion_analyzer(str(sys.argv[1]), output=False)  
            
        # dir内の全てのcsvファイルを分析 
        else:  
            dir = 'csv2/'                              
            target = {1:'priA', 2:'sesA', 3:'bothA', 4:'priB', 5:'sesB', 6:'bothB', 7:'category'}

            for target_type in range(7,8): # ここの範囲を調整して使う
                print(target[target_type])
                timeNames, df_emotion, df_category = create_ratioDF(target_type, output=False)
                plot_data(target_type)
                print()
            
