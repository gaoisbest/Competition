# -*- coding: utf-8 -*-

import fasttext as ft
import os


file_path = '/data/dse/lib/cassandra/AI100/'

input_file = os.path.join(file_path, 'fasttext_training_file.txt')
output = os.path.join(file_path, 'ai100_classification.model')
test_file = input_file

# set params
dim = 10
lr = 0.1 # 0.1
epoch = 100 # 5
min_count = 2
word_ngrams = 5
bucket = 1000000 # 10000000  1000000
thread = 4
silent = 1
label_prefix = '__label__'

print 'training---------'
# Train the classifier
classifier = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
    min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
    thread=thread, silent=silent, label_prefix=label_prefix)

print 'testing---------------'
# Test the classifier
result = classifier.test(test_file)
print 'P@1:', result.precision
print 'R@1:', result.recall
print 'Number of examples:', result.nexamples

