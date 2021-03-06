### GiNZA==5.1.0 ###
# coding:utf-8
import sys
args = sys.argv

import pandas as pd
import spacy
nlp = spacy.load('ja_ginza')

def tokenize():
  with open(args[1], 'w') as f:
    for line in lines:
      doc = nlp.tokenizer(line)

      for token in doc:
          f.write(token.orth_+' ')

      f.write('\n') 

def lemmatize():
  with open(args[1], 'w') as f:    
    for line in lines:
      doc = nlp.tokenizer(line)

      for token in doc:
          f.write(token.lemma_+' ')

      f.write('\n')

def POS():
  with open(args[1], 'w') as f:    
    for line in lines:
      doc = nlp.tokenizer(line)

      for token in doc:
          f.write(token.tag_+' ')

      f.write('\n')

def tokenPOS():
  with open(args[1], 'w') as f:
    for line in lines:
      doc = nlp.tokenizer(line)

      for token in doc:
        f.write(token.pos_+' '+token.orth_+'|')

      f.write('\n')

def lemmaPOS():
  with open(args[1], 'w') as f:
    for line in lines:
      doc = nlp.tokenizer(line)

      for token in doc:
          f.write(token.pos_+' '+token.lemma_+'|')
          
      f.write('\n')

def dependency():
  with open(args[1], 'w') as f:
    for line in lines:
        doc = nlp(line)

        for token in doc:
            f.write(token.dep_+' '+token.orth_+'|')

        f.write('\n')
        

def change_NER(x_train):

  result = []

  for line in x_train:
      sentence = ''
      text = list(line)
      doc = nlp(line)

      entity = [ent.label_ for ent in doc.ents]       # 固有表現のラベル
      start = [ent.start_char for ent in doc.ents]    # 何文字目から固有表現か
      end = [ent.end_char for ent in doc.ents]        # 何文字目まで固有表現か
      num = 0                                        
      nowNER = False
      
      for i in range(len(text)):                      # 1文字ずつループ
          
          if (len(start) != 0) and (i == start[num]): # 固有表現の開始位置に来たら
              sentence += entity[num]                 # 固有表現を追加
              if num < len(start) - 1:                # out of rangeの防止
                  num += 1
              nowNER = True

          elif nowNER == True:                        # 固有表現を認識したら
              if i < end[num-1]:                      # その文字数を消費
                  continue
              elif i == end[num-1]:
                  nowNER = False
                  sentence += text[i]

          else:
              sentence += text[i]
      
      result.append(sentence)


  return result


def write_NER():

  with open(args[1], 'r') as f:  
    lines = f.read().splitlines()

  with open(args[1], 'w') as f:
    lines = change_NER(lines)
    
    for sentence in lines:
      f.write(sentence+'\n')
    


def print_usage():
    print()

    print('usage : Command line arguments are supported as follows')
    print()
    print('1: tokenize')
    print('2: lemmatize')
    print('3: POS')
    print('4: tokenPOS')
    print('5: lemmaPOS')
    print("6: chunks(but this option is currently not supported)")
    print('7: dependency')
    print('8: chunksNER(You will need to prepare the chunked file in advance)')
    print('9: depNER')
    print('10: tokenNER')
    print('11: lemmaNER')

    print()   
    


if __name__ == '__main__':

  if 'txt' in args[1]:
    lines = open(args[1],'r').read().splitlines()
    
  else:
    isChange = True
    
    if 'tsv' in args[1]:
      df = pd.read_csv(args[1], sep='\t')
    elif 'csv' in args[1]:
      df = pd.read_csv(args[1])
    elif 'xlsx' in args[1]:
      df = pd.read_excel(args[1])

    lines = df.sentence.values.tolist()
    args[1] = 'result.txt'


  if args[2] == '1': 
    tokenize()
  elif args[2] == '2':
    lemmatize()
  elif args[2] == '3':
    POS()
  elif args[2] == '4':
    tokenPOS()
  elif args[2] == '5':
    lemmaPOS()
  elif args[2] == '7':
    dependency()
  elif args[2] == '9':
    dependency()
    write_NER()
  elif args[2] == '10':
    tokenize()
    write_NER()
  elif args[2] == '11':
    lemmatize()
    write_NER()


  else:
    print_usage()


  if isChange:
    lines = open('result.txt').read().splitlines()
    df['Text'] = lines
    df.to_csv('result.csv', index=False)
