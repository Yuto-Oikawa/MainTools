# encoding:utf-8
import MeCab

with open('#地震.txt','r') as f:
    data = f.read()
mecab = MeCab.Tagger('-Owakati')
text = mecab.parse(data)
mecab.parse('')

out_file_name = 'tokenized.txt'
with open(out_file_name, 'w') as f:
    f.write(text)