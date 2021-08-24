# coding:utf-8
import os
import shutil
from datetime import timedelta
import pandas as pd



def create_train_csv():
    # txtファイルをcsvファイルに変換する関数

    DIR = "train/"
    groups = [f for f in os.listdir(DIR)]
    groups = [f for f in groups if os.path.isfile(os.path.join(dir, f))]

    groups[0], groups[1], groups[2] = groups[1], groups[2], groups[0]
    train_all  = []
    length = []

    for group in groups:
        lines = open(DIR+str(group)).readlines()
        data = [line for line in lines]
        train_all.extend(data)
        length.append(len(data))


    sesquiary = length[0]
    primary = length[1]
    secondary = length[2]

    # typed = []
    # typed.extend(['t'] * 350)　# 台風
    # typed.extend(['g'] * 350)　# 豪雨
    # typed.extend(['f'] * 350)　# 福島ら


    df = pd.DataFrame()
    # df['Text'] = train_all
    # df['label'] = [0] * primary + [1] * secondary + [2] * sesquiary
    # df['type'] = typed * 3

    df1 = pd.read_excel('台風豪雨福島_binary_sec.xlsx')
    df2 = pd.read_excel('台風豪雨福島_binary_reserve_sec.xlsx')
    df3 = pd.concat([df1, df2])

    df3.to_csv('result.csv', index=False)


def split_csv_by_time(csvName:str):
    try:
        # [:-4]は/csv/の除去
        os.makedirs(dir[:-4] + csvName)
    except FileExistsError:
        # フォルダの中身を全て削除してから新規作成
        shutil.rmtree(dir[:-4] + csvName)
        os.makedirs(dir[:-4]+ csvName)

    df = pd.read_csv(dir + csvName)
    df = df.dropna()
    INDEX = len(df)-1
    HOURS = 6

    df['time'] = pd.to_datetime(df['time'])
    base_time = df[-1:]['time'][INDEX]
    end_time = base_time + timedelta(hours=HOURS)
    max_time = df[1:]['time'][1]
    
    num = 1
    while base_time < max_time:
        df_split = df.query('@base_time <= time <= @end_time')
        
        # [-8:-3]は年月日秒の除去(時分のみ)
        base_name = str(base_time)[-8:-3]
        end_name = str(end_time)[-8:-3]
        length = len(df_split)
        df_split.to_csv(f'{dir[:-5]}/{csvName}/{num}_{base_name}_{end_name}_{length}件.csv', index=False)
        
        base_time = base_time + timedelta(hours=HOURS)
        end_time = end_time + timedelta(hours=HOURS)
        num += 1
        

if __name__ == '__main__':
    # create_train_csv()

    dir_name = ['#大雨/csv/', '#豪雨/csv/', '#大雨特別警報/csv/', '#線状降水帯/csv/', ]
    dir_name2 = ['#秋雨前線/csv/', '#洪水/csv/',  '#非常に激しい雨/csv/', '#猛烈な雨/csv/']
    dir_name.extend(dir_name2)
    
    for dir in dir_name:
        files = [f for f in os.listdir(dir)]
        files = [f for f in files if os.path.isfile(os.path.join(dir, f))]

        try:
            files.remove('.DS_Store')
        except:
            pass
    
        for csvName in sorted(files):
            split_csv_by_time(str(csvName))