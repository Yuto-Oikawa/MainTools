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

FILE_NAME = 'kappa_data/kappa_first_annotation.xlsx'
df = pd.ExcelFile(FILE_NAME)
sheet_names = df.sheet_names


def calcKappa(sheet_name:str, label=None,detail=False):
    df = pd.read_excel(FILE_NAME, sheet_name=sheet_name)
    
    if detail == False:
        y1 = (df['label'] == label)
        y2 = (df['others_label'] == label)
        if label == -1:     # 全カテゴリの場合
            y1 = df.label.values.tolist()
            y2 = df.others_label.values.tolist()
            
    elif detail == True:
        y1 = df.detail.values.tolist()
        y2 = df.others_detail.values.tolist()

    k = cohen_kappa_score(y1, y2)
    return k
    
    
def calcKappa_All(reverse=False):
    y1_list = []
    y2_list = []

    for sheet_name in sheet_names:
        df = pd.read_excel(FILE_NAME, sheet_name=sheet_name)
        y1 = df.label.values.tolist()
        y2 = df.others_label.values.tolist()
        y1_list.extend(y1)
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


def calcKappa_subCategory(sheet_name:str, detail:str, label:int):

    df = pd.read_excel(FILE_NAME, sheet_name=sheet_name)
    detail1 = (df['detail'] == detail)
    detail2 = (df['others_detail'] == detail)
    df_Mydetail = df[detail1]
    df_Yourdetail = df[detail2]
        
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

    print(f"平均値: {avg:.3f}")
    print(f"最大値: {max:.3f}")
    print(f"最小値: {min:.3f}")
    print(f"標準偏差: {sd:.3f}")


def printKappa(detail=False):
    # calcKappa()の1つまたは2つの引数をfor文で回す
    
    if detail == True:                  # サブカテゴリのKappa
        all = []
        for sheet_name in sheet_names:
            kappa = calcKappa(sheet_name, detail=True)
            print(f'{sheet_name}, {kappa:.3f}')
            all.append(kappa)
        if len(all) > 2:
            print()
            print_statistic(all)
            print()
            
    
    elif detail == False:               # カテゴリのKappa
        label_name = {1:'1次情報', 0:'1.5次情報', 2:'2次情報', -1:'全カテゴリ'}
        
        print()
        for sheet_name in sheet_names:
            print(sheet_name)
            all = []
            
            for label in label_name.keys():
                kappa = calcKappa(sheet_name, label, detail=False)
                print(f"{label_name[label]}:{kappa:.3f}")
                all.append(kappa)
            print()
            
            if len(all) > 2:
                print_statistic(all)
                print()
    

def printKappa_subCategory(dataNameMain=True):
    # calcKappa_subCategory()の3つの引数をfor文で回す
    
    if dataNameMain == True:
        for sheet_name in sheet_names:         
            all = []
            print(sheet_name)                           # 対象となるデータを
            print('-'*30)

            for label, list in enumerate(detail_list.values()):
                for detail in list.keys():              # サブカテゴリごとに分析
                    k, meYour_error, yourMe_error = calcKappa_subCategory(sheet_name, detail, label)
                    print(list[detail]+" {:.3f}, 及川{}件, 他者{}件".format(k,len(meYour_error),len(yourMe_error)))
                    all.append(k)
                print('-'*30)
                    
            print_statistic(all)
            print('-'*30)



    elif dataNameMain == False:
        for label, list in enumerate(detail_list.values()):
            for detail in list.keys():
                all = []
                sumA = 0
                sumB = 0
                print('-'*30)
                print(list[detail])                              # 対象となるサブカテゴリを
                
                for sheet_name in sheet_names:                   # データごとに分析  
                    k, meYour_error, yourMe_error = calcKappa_subCategory(sheet_name, detail, label)
                    all.append(k)
                    
                    print(sheet_name+" {:.3f}, 及川{}件, 他者{}件".format(k,len(meYour_error),len(yourMe_error)))
                    sumA += len(meYour_error)
                    sumB += len(yourMe_error)
                print('-'*30)
                    
                if len(all) > 2:
                    print_statistic(all)
                    print('計:及川{}件，他者{}件'.format(sumA,sumB))



if __name__ == '__main__':
    listA = {'ik':'意見', 'k':'感情表現', 'is':'意思表示', 'y':'呼びかけ', 's':'その他'}
    listB = {'t':'体験的事実', 'j':'発生的事実', 'd':'言及的事実'}
    listC = {'ds':'伝聞推定', 'u':'ニュース記事'}
    detail_list = {0:listA, 1:listB, 2:listC}


    if args.all:
        calcKappa_All(reverse=False)
    elif args.all2:
        calcKappa_All(reverse=True)
        
    elif args.subCategory:
        printKappa_subCategory(dataNameMain=True)
    elif args.subCategory2:
        printKappa_subCategory(dataNameMain=False)
        
    elif args.detail:
        printKappa(detail=True)
    else:
        printKappa(detail=False)
