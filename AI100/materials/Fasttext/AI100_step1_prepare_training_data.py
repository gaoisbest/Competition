# -*- coding: utf-8 -*-
import pandas as pd
import jieba as jb
import codecs
import os

def read_file(file_path):
    f = codecs.open(file_path, encoding='utf-8')
    lines = []
    for line in f:
        line = line.rstrip('\n').rstrip('\r')
        lines.append(line)
    return lines

def cut_content(each_row):
    return ' '.join([word for word in jb.lcut(each_row['text_content']) if word not in stopwordsCN])





file_path = '/data/dse/lib/cassandra/AI100/'

stopwordsCN = read_file(os.path.join(file_path, 'stopWords_cn.txt'))

training_data = pd.read_csv(os.path.join(file_path, 'training.csv'), names=['text_label', 'text_content'], encoding='utf8')

training_data['text_content_segmentation'] = training_data.apply(lambda row: cut_content(row), axis=1)

training_file = codecs.open(file_path + 'fasttext_training_file.txt', 'w', 'utf-8')
c = 0
for index, row in training_data.iterrows():
    tmp_str = '__label__' + str(row['text_label']) + ' , ' + row['text_content_segmentation']
    training_file.write(tmp_str + '\n')
    c += 1
    print c
training_file.close()



