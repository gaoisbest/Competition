# -*- coding: utf-8 -*-
import fasttext as ft
import pandas as pd
import jieba as jb
import os
import codecs


def read_file(file_path):
    f = codecs.open(file_path, encoding='utf-8')
    lines = []
    for line in f:
        line = line.rstrip('\n').rstrip('\r')
        lines.append(line)
    return lines

file_path = '/data/dse/lib/cassandra/AI100/'

label_prefix = '__label__'

def predict_row_label(each_row):
    tmp_list = [each_row['text_content_segmentation']]
    label = fasttext_classifier.predict(tmp_list)
    return label[0][0]

def cut_content(each_row):
    return ' '.join([word for word in jb.lcut(each_row['text_content']) if word not in stopwordsCN])


file_path = '/data/dse/lib/cassandra/AI100/'
stopwordsCN = read_file(os.path.join(file_path, 'stopWords_cn.txt'))

testing_data = pd.read_csv(os.path.join(file_path, 'testing.csv'), names=['text_index', 'text_content'])
testing_data['text_content_segmentation'] = testing_data.apply(lambda row: cut_content(row), axis=1)

fasttext_classifier = ft.load_model(os.path.join(file_path, 'ai100_classification.model.bin'), label_prefix=label_prefix)
testing_data['predicted_label'] = testing_data.apply(lambda row: predict_row_label(row), axis=1)

header = ['text_index', 'predicted_label']
testing_data.to_csv(os.path.join(file_path, 'results.csv'), encoding='utf-8', columns=header, index=False, header=False)
