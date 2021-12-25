import os
import pandas as pd
import openpyxl
from pandas.core.algorithms import mode
from torch.autograd.grad_mode import F

FILE_NAME = 'data.xlsx'
DIR = 'files/'
res_list, n1_list, n2_list, n3_list, n4_list = [],[],[],[],[]


def get_file_name():
    files = [f for f in os.listdir(DIR)]
    files = sorted(files)
    try:
        files.remove('.DS_Store')
    except:
        pass
    
    return files


def append_list(lines):
    for i in range(len(lines)):
        elements = lines[i].split()        
        res_list.append(elements[0])
        n1_list.append(elements[1])
        
        
        if len(elements) == 2:
            n2_list.append(0)
            continue
        
        n2_list.append(elements[2])
        # n3_list.append(elements[3])
        # n4_list.append(elements[4])


def create_excel(files):
    
    book = openpyxl.Workbook()
    book.save(FILE_NAME)
    
    with pd.ExcelWriter(FILE_NAME, mode='a') as writer:
        
        for sheet_name in files:
            print(sheet_name)
            df = pd.DataFrame()
            lines = open(DIR+sheet_name).readlines()
            append_list(lines)
            
            df['n1_list'] = n1_list   
            df['n2_list'] = n2_list 
            # df['n3_list'] = n3_list 
            df['res_list'] = res_list 
            # df['n4_list'] = n4_list
            
            df.columns=['n1','n2','res']
            sheet_name = sheet_name.replace('sm', '').replace('_res.txt', '')
            df.to_excel(writer, sheet_name=sheet_name)
            
            n1_list.clear()
            n2_list.clear()
            res_list.clear()


    workbook = openpyxl.load_workbook(filename=FILE_NAME)
    workbook.remove(workbook['Sheet'])
    workbook.save(FILE_NAME)


if __name__ == '__main__':
    files = get_file_name()
    create_excel(files)
