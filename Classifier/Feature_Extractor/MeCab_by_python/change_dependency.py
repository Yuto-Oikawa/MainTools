# encoding: utf-8

import re
import xml.etree.ElementTree as ET
from tabulate import tabulate
from pprint import pprint

class Morph:
    def __init__(self, token):
        self.surface = token.text
        feature = token.attrib['feature'].split(',')
        self.base = feature[6]
        self.pos = feature[0]
        self.pos1 = feature[1]

    def __repr__(self):
        return self.surface

class Chunk(list):
    def __init__(self, chunk):
        self.morphs =  [Morph(morph) for morph in chunk]
        super().__init__(self.morphs)
        self.dst = int(chunk.attrib['link'])
        self.srcs = []
    def __repr__(self):
        return re.sub(r'[、。]', '', ''.join(map(str, self)))

class Sentence(list):
    def __init__(self, sent):
        self.chunks = [Chunk(chunk) for chunk in sent]
        super().__init__(self.chunks)
        for i, chunk in enumerate(self.chunks):
            if chunk.dst != -1:
                self.chunks[chunk.dst].srcs.append(i)


with open('data/primary-all.xml.cabocha') as f:
    root = ET.fromstring("<sentences>" + f.read() + "</sentences>")
text = [Sentence(sent) for sent in root]

table = [
    [''.join([morph.surface for morph in chunk]), chunk.dst]
    for i in range(len(text))
        for chunk in text[i]
 ]
# result = tabulate(table, tablefmt = 'html', headers = ['番号', '文節', '係り先'], showindex = 'always')
with open('dependency.txt', 'w') as f:
    for i in range(len(table)):
        f.write(str(table[i][1])) # num
        f.write('D | ')
        f.write(str(table[i][0])) # word
        f.write(' ')