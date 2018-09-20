
# coding: utf-8

import numpy as np
import re
from tensorflow.contrib import learn
import math

def clean_str(string):
    # substitute space for characters not on the list
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # add space
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    # consecutive spaces are trimmed down to one space
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# returns: 
# x: data, list of strings 
# y: labels, list of 1s(positive) or 0s(negative)

def load_data_and_labels():
    positive_data_file_dir = "./data/rt-polaritydata/rt-polarity.pos"
    negative_data_file_dir = "./data/rt-polaritydata/rt-polarity.neg"


    infile = open(positive_data_file_dir, "r", encoding="utf-8")
    positive_examples = infile.readlines()
    positive_examples = [line.strip() for line in positive_examples]

    infile = open(negative_data_file_dir, "r", encoding="utf-8")
    negative_examples = infile.readlines()
    negative_examples = [line.strip() for line in negative_examples]
    
    x = positive_examples + negative_examples
    x = list(map(clean_str, x))
#     x = [clean_str(sent) for sent in x]
    
    positive_labels = [[1, 0]] * len(positive_examples)
    negative_labels = [[0, 1]] * len(negative_examples)
    y = positive_labels + negative_labels    
    return x, y

# arg:
# x: list of strings
# return:
# integerized (indexed) and padded version of x, in the form of np.ndarray
def index_and_pad(x):
    max_document_length = max([len(doc.split()) for doc in x])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    return np.array(list(vocab_processor.fit_transform(x))), max_document_length, vocab_processor.vocabulary_

# partition data into training set and development set according to dev_sample_percentage
def partition_data_and_labels(data, labels, dev_sample_percentage):
    assert len(data) == len(labels), "batch_iter: length of data doesn't equal length of labels"
    dev_sample_start = int(float(len(data)) * (1 - dev_sample_percentage))
    return data[:dev_sample_start], data[dev_sample_start:], labels[:dev_sample_start], labels[dev_sample_start:]

# shuffles a data labels pair, returns shuffled np.arrays
def shuffle_data_and_labels(data, labels):
    assert len(data) == len(labels), "shuffle: length of data doesn't equal length of labels"
    data = np.array(data)
    labels = np.array(labels)
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    return data[shuffle_indices], labels[shuffle_indices]

# batch generator
def batch_generator(data, labels, batch_size):
    assert len(data) == len(labels), "batch_iter: length of data doesn't equal length of labels"
    num_of_batches = math.ceil(float(len(data)) / float(batch_size))
    for i in range(num_of_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(data))
        yield data[start_index : end_index], labels[start_index : end_index]