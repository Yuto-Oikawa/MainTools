# coding:utf-8
import pandas as pd
import os


def create_train_csv():
    # txtファイルをcsvファイルに変換する関数です

    DIR = "train/"
    groups = [f for f in os.listdir(DIR)]
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
    #df['type'] = typed * 3

    df1 = pd.read_excel('台風豪雨福島_binary_sec.xlsx')
    df2 = pd.read_excel('台風豪雨福島_binary_reserve_sec.xlsx')
    df3= pd.concat([df1, df2])

    df3.to_csv('result.csv', index=False)


if __name__ == '__main__':
    create_train_csv()