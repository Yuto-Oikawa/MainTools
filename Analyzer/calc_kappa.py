import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--all', action='store_true')
parser.add_argument('-a2', '--all2', action='store_true')
parser.add_argument('-s', '--subCategory', action='store_true')
parser.add_argument('-s2', '--subCategory2', action='store_true')
parser.add_argument('-d', '--detail', action='store_true')
args = parser.parse_args()

df1 = pd.read_excel('kappa_data/heavy_rain.xlsx',sheet_name=[2,3,4])
df2 = pd.read_excel('kappa_data/typhoon_サブカテゴリ統一.xlsx',sheet_name=[0,1,2,3])


def calcKappa(sheetName:int, dataName:str, label=None,detail=False):
    df = df1 if dataName == 'heavy_rain' else df2
    
    if detail == False:
        y1 = df[sheetName].label.values.tolist()
        y2 = df[sheetName].others_label.values.tolist()
        detail1 = (df[sheetName]['label'] == label)
        detail2 = (df[sheetName]['others_label'] == label)
        
        if label == -1:     # 全カテゴリの場合
            detail1 = y1
            detail2 = y2
            
        k = cohen_kappa_score(detail1, detail2)
        print(name[dataName][sheetName]+": {:.3f}".format(k))    

    elif detail == True:
        y1 = df[sheetName].detail.values.tolist()
        y2 = df[sheetName].others_detail.values.tolist()
        k = cohen_kappa_score(y1, y2)
        print(name[dataName][sheetName]+": {:.3f}".format(k))
        
    return k
    
    
def calcKappa_All(reverse=False):
    y1_list = []
    y2_list = []

    for sheetName in range(3):
        y1 = df1[sheetName+2].label.values.tolist()
        y1_list.extend(y1)
        y2 = df1[sheetName+2].others_label.values.tolist()
        y2_list.extend(y2)

    for sheetName in range(4):
        y1 = df2[sheetName].label.values.tolist()
        y1_list.extend(y1)
        y2 = df2[sheetName].others_label.values.tolist()
        y2_list.extend(y2)

    print()
    print('All')
    k = cohen_kappa_score(y1_list, y2_list)
    print("k: {:.3f}".format(k))
    print(confusion_matrix(y1_list,y2_list))
    
    if reverse == False:
        print(classification_report(y1_list, y2_list, digits = 2,target_names=['1.5次情報','1次情報','2次情報']))
    elif reverse == True:
        print(classification_report(y2_list, y1_list, digits = 2,target_names=['1.5次情報','1次情報','2次情報']))


def calcKappa_detail(sheetName:int, dataName:str, detail:str, label:int):

    df = df1 if dataName == 'heavy_rain' else df2
    detail1 = (df[sheetName]['detail'] == detail)
    detail2 = (df[sheetName]['others_detail'] == detail)
    df_Mydetail = df[sheetName][detail1]
    df_Yourdetail = df[sheetName][detail2]
        
    k = cohen_kappa_score(detail1, detail2)
    meYour_error = df_Mydetail[df_Mydetail['others_label'] != label]    # 自分が付与したサブカテゴリで，他者とカテゴリが異なる場合
    yourMe_error = df_Yourdetail[df_Yourdetail['label'] != label]       # 他者が付与したサブカテゴリで，自分とカテゴリが異なる場合
    
    return k, meYour_error, yourMe_error



def print_statistic(all:list):
    all = np.array(all)
    avg = np.average(all)
    sd = np.std(all)
    min = np.min(all)
    max = np.max(all)

    print()
    print(f"平均値: {avg:.3f}")
    print(f"最大値: {max:.3f}")
    print(f"最小値: {min:.3f}")
    print(f"標準偏差: {sd:.3f}")
    print()


def printKappa(detail=False):
    # calcKappa()の2つまたは3つの引数をfor文で回す
    
    if detail == True:                  # サブカテゴリのKappa
        all = []
        for dataName in name.keys():
            for sheetName in name[dataName].keys():
                kappa = calcKappa(sheetName,dataName, detail=True)
                all.append(kappa)
        print_statistic(all)
    
    elif detail == False:               # カテゴリのKappa
        label_name = {0:'1.5次情報', 1:'1次情報', 2:'2次情報', -1:'全カテゴリ'}
        
        for label in label_name.keys():
            all = []
            print(label_name[label])                    # カテゴリごとに分析
            
            for dataName in name.keys():                # 台風または豪雨の
                for sheetName in name[dataName].keys(): # 対象となるデータ
                    kappa = calcKappa(sheetName, dataName, label,detail=False)
                    all.append(kappa)
                    
            print_statistic(all)
    

def printKappa_subCategory(dataNameMain=False):
    # calcKappa_detail()の4つの引数をfor文で回す
    
    if dataNameMain == True:
        for  dataName in name.keys():                       # 台風または豪雨の
            for sheetName in name[dataName].keys():         
                all = []
                print()
                print(name[dataName][sheetName])            # 対象となるデータを

                for label, list in enumerate(detail_list.values()):
                    for detail in list.keys():              # サブカテゴリごとに分析
                        k, meYour_error, yourMe_error = calcKappa_detail(sheetName, dataName, detail, label)
                        print(list[detail]+" {:.3f}, 及川{}件, 他者{}件".format(k,len(meYour_error),len(yourMe_error)))
                        all.append(k)
                        
                print_statistic(all)

    elif dataNameMain == False:
        for label, list in enumerate(detail_list.values()):
            for detail in list.keys():
                all = []
                sumA = 0
                sumB = 0
                print()
                print(list[detail])                              # 対象となるサブカテゴリを
                
                for dataName in name.keys():                     # 台風または豪雨の
                    for sheetName in name[dataName].keys():      # データごとに分析  
                        k, meYour_error, yourMe_error = calcKappa_detail(sheetName, dataName, detail, label)
                        all.append(k)
                        
                        print(name[dataName][sheetName]+" {:.3f}, 及川{}件, 他者{}件".format(k,len(meYour_error),len(yourMe_error)))
                        sumA += len(meYour_error)
                        sumB += len(yourMe_error)
                
                print_statistic(all)
                print('計:及川{}件，他者{}件'.format(sumA,sumB))
                print()



if __name__ == '__main__':
    # 設定パラメータ
    # ここを変更して使う
    name = {}
    name['heavy_rain'] = {2:'0706', 3:'0707', 4:'0708'}
    name['typhoon'] = {0:'9A', 1:'9B', 2:'10A', 3:'10B'}
    listA = {'ik':'意見', 'k':'感情表現', 'is':'意思表示', 'y':'呼びかけ', 's':'その他'}
    listB = {'t':'直接体験', 'j':'事実', 'd':'断定表現'}
    listC = {'ds':'伝聞推定', 'u':'ニュース記事'}
    detail_list = {0:listA, 1:listB, 2:listC}


    if args.all:
        calcKappa_All(reverse=False)
    elif args.all2:
        calcKappa_All(reverse=True)
        
    elif args.subCategory:
        printKappa_subCategory(dataNameMain=False)
    elif args.subCategory2:
        printKappa_subCategory(dataNameMain=True)
        
    elif args.detail:
        printKappa(detail=True)
    else:
        printKappa(detail=False)