# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
import pandas as pd
import os
import jieba as jb
import codecs
import numpy as np
from sklearn.linear_model import SGDClassifier

def read_file(file_path):
    f = codecs.open(file_path, encoding='utf-8')
    lines = []
    for line in f:
        line = line.rstrip('\n').rstrip('\r')
        lines.append(line)
    return lines

def cut_content(each_row):
    return ' '.join([word for word in jb.lcut(each_row['text_content']) if word not in stopwordsCN])


def get_sentence_vector(sen, dim):
    sen_vec = np.zeros(dim).reshape((1, dim))
    cc = 0
    for word in sen:
        try:
            sen_vec += word2vec_model[word].reshape((1, dim))
            cc += 1
        except Exception:
            pass
    if cc != 0:
        sen_vec /= cc
    return sen_vec

def get_data_sentences(input_data):
    rtn_sentence = []
    cc = 0
    for curr_index, curr_row in input_data.iterrows():
        rtn_sentence.append(curr_row['text_content_segmentation'].split())
        cc += 1
    return rtn_sentence, cc

def get_sentence_vectors(data_sentences, model_size):
    sentences_vector = []
    cc = 0
    for each_sen in data_sentences:
        sentences_vector.append(get_sentence_vector(each_sen, model_size))
        cc += 1
    return sentences_vector, cc

if __name__ == '__main__':
    data_dir = '/data/dse/lib/cassandra/AI100/'
    stopwordsCN = read_file(os.path.join(data_dir, 'stopWords_cn.txt'))

    training_data = pd.read_csv(os.path.join(data_dir, 'training.csv'), names=['text_label', 'text_content'], encoding='utf8')
    training_data['text_content_segmentation'] = training_data.apply(cut_content, axis=1)

    training_sentences, training_count = get_data_sentences(training_data)
    print 'training sentence counts:{}'.format(training_count)

    model_size = 10
    word2vec_model = Word2Vec(training_sentences, size=model_size, min_count=3)

    # save the model
    # word2vec_model.save(os.path.join(data_dir, 'wor2vec_model.model'))
    # load model
    # word2vec_model = Word2Vec.load(os.path.join(data_dir, 'wor2vec_model.model'))

    training_sentences_vector, training_count = get_sentence_vectors(training_sentences, model_size)
    print 'training sentence counts:{}'.format(training_count)


    svm_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=20)
    svm_clf.fit(np.concatenate(training_sentences_vector),  training_data['text_label'])


    testing_data = pd.read_csv(os.path.join(data_dir, 'testing.csv'), names=['text_index', 'text_content'])
    testing_data['text_content_segmentation'] = testing_data.apply(cut_content, axis=1)

    testing_sentences, testing_count = get_data_sentences(testing_data)
    print 'testing sentence counts:{}'.format(testing_count)

    testing_sentences_vector, testing_count = get_sentence_vectors(testing_sentences, model_size)
    print 'testing sentence counts:{}'.format(testing_count)

    tested_predicted = []
    for each_sen_vec in testing_sentences_vector:
        pre_res = svm_clf.predict(each_sen_vec)
        tested_predicted.append(pre_res[0])

    tested_predicted = pd.DataFrame({'predicted_label': tested_predicted})
    np.savetxt(os.path.join(data_dir, 'word2vec_svm_results.csv'),
               np.dstack((np.arange(1, testing_count + 1), tested_predicted))[0], "%d,%d")