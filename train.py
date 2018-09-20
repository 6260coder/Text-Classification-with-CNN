
# coding: utf-8


import tensorflow as tf
import numpy as np
import math
from text_cnn import TextCNN
import data_helpers

# prepare data
dev_sample_percentage = 0.1
x, y = data_helpers.load_data_and_labels()
x, max_doc_length, vocab = data_helpers.index_and_pad(x)
x, y = data_helpers.shuffle_data_and_labels(x, y)
data_train, data_dev, labels_train, labels_dev = data_helpers.partition_data_and_labels(x, y, dev_sample_percentage)

# create model
num_of_classes = 2
vocab_size = len(vocab)
embedding_dimension = 128
filters_config = [(3, 64), (4, 64), (5, 64)]

cnn = TextCNN(max_doc_length, 
              num_of_classes, 
              vocab_size, 
              embedding_dimension, 
              filters_config, 
              l2_reg_lambda=0.0005)

# train model
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cnn.loss)
# optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(cnn.loss)

def test():
    feed_dict = {cnn.docs: data_dev, cnn.labels : labels_dev, cnn.dropout_keep_prob : 1.0}
    accuracy, loss = sess.run([cnn.accuracy, cnn.loss], feed_dict=feed_dict)
    print("- accuracy:{0}, loss:{1}".format(accuracy, loss))

def test_1():
    feed_dict = {cnn.docs: data_train[:1000], cnn.labels : labels_train[:1000], cnn.dropout_keep_prob : 1.0}
    accuracy, loss = sess.run([cnn.accuracy, cnn.loss], feed_dict=feed_dict)
    print("- accuracy:{0}, loss:{1}".format(accuracy, loss))
    
def test_2():
    feed_dict = {cnn.docs: data_train, cnn.labels : labels_train, cnn.dropout_keep_prob : 1.0}
    accuracy, loss = sess.run([cnn.accuracy, cnn.loss], feed_dict=feed_dict)
    print("- accuracy:{0}, loss:{1}".format(accuracy, loss))

batch_size = 64
init = tf.global_variables_initializer()
# num_of_batches = math.ceil(float(len(data_train)) / float(batch_size))
with tf.Session() as sess:
    sess.run(init)
    test_1()
    for epoch in range(1000):
        data_train, labels_train = data_helpers.shuffle_data_and_labels(data_train, labels_train)
        print("epoch{}:".format(epoch))
        current_batch = 0
        batch_generator = data_helpers.batch_generator(data_train, labels_train, batch_size)
        for batch_data, batch_labels in batch_generator:
            if current_batch % 100 == 0: pass
            feed_dict = {cnn.docs : batch_data, cnn.labels : batch_labels, cnn.dropout_keep_prob : 0.7}
            sess.run(optimizer, feed_dict=feed_dict)
            current_batch += 1
        test_1()