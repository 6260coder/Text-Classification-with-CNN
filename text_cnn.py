

import tensorflow as tf
import numpy as np
import math

class TextCNN(object):
    def __init__(self,
                 max_doc_length,
                 num_of_classes,
                 vocab_size,
                 embedding_dim,
                 filters_config, 
                 l2_reg_lambda=0.0): 
        with tf.name_scope("inputs"):
            self.docs = tf.placeholder(dtype=tf.int32, 
                                       shape=[None, max_doc_length], 
                                       name="docs")
            self.labels = tf.placeholder(dtype=tf.int32, 
                                         shape=[None, num_of_classes], 
                                         name="labels")
            self.dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                    shape=[],
                                                    name="dropout_keep_prob")
            
        with tf.name_scope("embedding"):
            self.vocab_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), 
                                                name="vocab_embeddings")
            self.embedded_docs = tf.nn.embedding_lookup(self.vocab_embeddings, self.docs)
        
        with tf.name_scope("convolution"):
            num_filters_total = 0
            max_pooled_results = []
            # The added dimension represents num of channels, which is 1 in this case
            # as usually is in NLP problems.
            self.embedded_docs = tf.expand_dims(self.embedded_docs, -1)
            for filter_size, num_of_filters in filters_config:
                num_filters_total += num_of_filters
                with tf.name_scope("{}-gram_filters".format(filter_size)):
                    filter_shape = [filter_size, embedding_dim, 1, num_of_filters]
                    Filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter")
                    bias = tf.Variable(tf.constant(0.1, shape=[num_of_filters]), name="bias")
                    conved = tf.nn.conv2d(input=self.embedded_docs,
                                        filter=Filter,
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="conv") + bias
                    conv = tf.nn.relu(conved, name="relu")
                    pooled = tf.nn.max_pool(value=conved,
                                            ksize=[1, max_doc_length - filter_size + 1, 1, 1], 
                                            strides=[1, 1, 1, 1], 
                                            padding="VALID", 
                                            name="max_pool")
                    max_pooled_results.append(pooled)
            # rehape to get rid of the superfuous dims (vertical_strides and horizontal_strides,
            # which are both 1 at this point
            self.features = tf.reshape(tf.concat(max_pooled_results, -1), 
                                       [-1, num_filters_total])

        with tf.name_scope("drop_out"):
            self.features_drop_out = tf.nn.dropout(self.features, self.dropout_keep_prob)
        
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([int(self.features.shape[-1]), num_of_classes], stddev=0.1), 
                            name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_of_classes]), 
                            name="b")
            self.scores = tf.nn.xw_plus_b(self.features_drop_out, 
                                          W, 
                                          b, 
                                          name="scores")
            self.predictions = tf.argmax(self.scores, -1, name="predictions")
        with tf.name_scope("loss"):
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, 
                                                                            labels=self.labels)
            cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
            regularization_loss = l2_reg_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))
            self.loss = cross_entropy_loss + regularization_loss            
            
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")