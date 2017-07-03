# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import jieba as jb
import codecs
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

def read_file(file_path):
    """
    read the input file encoding by utf-8
    :param file_path: input file path
    :return: a list contain all lines of the input file
    """
    f = codecs.open(file_path, encoding='utf-8')
    lines = []
    for line in f:
        line = line.rstrip('\n').rstrip('\r')
        lines.append(line)
    return lines


def seg_content(each_row):
    """
    word segmentation 
    :param each_row: the un-segmented text
    :return: segmented text by space
    """
    return ' '.join([word for word in jb.lcut(each_row['text_content']) if word not in stopwordsCN])

if __name__ == '__main__':
    # directory of all material
    data_dir = './materials'
    stopwordsCN = read_file(os.path.join(data_dir, 'stopWords_cn.txt'))
    new_words = read_file(os.path.join(data_dir, 'ai100_words.txt'))

    for word in new_words:
        jb.add_word(word)

    training_data = pd.read_csv(os.path.join(data_dir, 'training.csv'), names=['text_label', 'text_content'], encoding='utf-8')
    training_data['text_content_segmentation'] = training_data.apply(seg_content, axis=1)


    text_clf = Pipeline([('word_counter', CountVectorizer(ngram_range=(1, 2), min_df=1)),
                    ('tfidf_computer', TfidfTransformer(smooth_idf=True, sublinear_tf=True)),
                    ('clf', SGDClassifier(loss='hinge', n_iter=20, penalty='elasticnet', alpha=1e-5))])

    text_clf.fit(training_data['text_content_segmentation'], training_data['text_label'])

    testing_data = pd.read_csv(os.path.join(data_dir, 'testing.csv'), names=['text_index', 'text_content'])
    testing_data['text_content_segmentation'] = testing_data.apply(seg_content, axis=1)
    tested_predicted = text_clf.predict(testing_data['text_content_segmentation'])
    np.savetxt(os.path.join(data_dir, 'bow_svm_results.csv'), np.dstack((np.arange(1, tested_predicted.size+1), tested_predicted))[0],"%d,%d")


