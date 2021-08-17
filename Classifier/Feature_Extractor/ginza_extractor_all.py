### GiNZA==3.1.2 ###

# coding:utf-8
import spacy
import sys

def change_phase1():

    with open(args[1],'r') as f:
        lines = f.read().splitlines()

        with open('01_tokenized.txt', 'w') as f1, open('02_lemmatized.txt', 'w') as f2, open('03_pos.txt', 'w') as f3, \
        open('04_tokenPOS.txt', 'w') as f4, open('05_lemmaPOS.txt', 'w') as f5:

            for line in lines:
                doc = nlp.tokenizer(line)

                for token in doc:
                    f1.write(token.orth_+' ')
                    f2.write(token.lemma_+' ')
                    f3.write(token.tag_+' ')
                    f4.write(token.pos_)
                    f4.write(' ')
                    f4.write(token.orth_)
                    f4.write('|')
                    f5.write(token.pos_)
                    f5.write(' ')
                    f5.write(token.lemma_)
                    f5.write('|')
                f1.write('\n')
                f2.write('\n')                
                f3.write('\n')
                f4.write('\n')            
                f5.write('\n')

        with open('07_dependency.txt', 'w') as f:
            for line in lines:
                doc = nlp(line)

                for token in doc:
                    f.write(token.dep_)
                    f.write(' ')
                    f.write(token.orth_)
                    f.write('|')

                f.write('\n')


def NER_extract(write, read):
    with open(write, 'w') as f1:

        with open(read, 'r', encoding='utf-8') as f2:
            lines2 = f2.read().splitlines()

            for line in lines2:
                text = list(line)
                doc = nlp(line)

                entity = [ent.label_ for ent in doc.ents]       # 固有表現のラベル
                start = [ent.start_char for ent in doc.ents]    # 何文字目から固有表現か
                end = [ent.end_char for ent in doc.ents]        # 何文字目まで固有表現か
                num = 0                                        
                stop = False

                for i in range(len(text)):
                    if (len(start) != 0) and (i == start[num]):
                        f1.write(entity[num])
                        if num < len(start) - 1:                # out of rangeの防止
                            num += 1
                        stop = True

                    elif stop == True:
                        if i < end[num-1]:
                            continue
                        elif i == end[num-1]:
                            stop = False
                            f1.write(text[i])

                    else:
                        f1.write(text[i])
                    
                f1.write('\n')




if __name__ == "__main__":
    args = sys.argv
    nlp = spacy.load('ja_ginza')

    change_phase1()

    NER_extract('08_chunksNER.txt',('06_chunks.txt'))
    NER_extract('09_depNER.txt',   ('07_dependency.txt'))
    NER_extract('10_tokenNER.txt', ('01_tokenized.txt'))
    NER_extract('11_lemmaNER.txt', ('02_lemmatized.txt'))    
