from SentimentAnalysis.model_params import params
import tensorflow as tf
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import gensim
import pandas as pd
import gzip
from tqdm import tqdm
import ipdb
import os
import random
import shutil
from SentimentAnalysis.my_model import DataClass

from gensim.models.wrappers import FastText
# import data_precessing as Data
max_word_count = 20
global data_split
data_split = [0.8, 0.1, 0.1]
train = False

# data_x = tf.layers.batch_normalization(data_x, training=(not predict))


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def get_sent(vectors):
    word_list = []
    for i in vectors:
        if (i != np.zeros([300])).all():
            word_list.append(
                data_class.vec_model.similar_by_vector(i)[0][0])
    return " ".join(word_list)


def create_lstm_cell(units):
    # lstmCell = tf.contrib.rnn.LayerNormBasicLSTMCell(
    #     units, activation=tf.nn.sigmoid, dropout_keep_prob=params.dropout_keep_prob,)
    lstmCell = tf.nn.rnn_cell.LSTMCell(
        units, activation=tf.nn.tanh, initializer=tf.contrib.layers.xavier_initializer())
    lstmCell = tf.nn.rnn_cell.DropoutWrapper(
        lstmCell, params.dropout_keep_prob)
    # lstmCell = tf.contrib.rnn.AttentionCellWrapper(lstmCell, 5)
    return lstmCell


def build_model(data_x):
    data_x_tri = tf.layers.conv1d(data_x, filters=150, kernel_size=[
        3], padding="same", activation=tf.nn.elu)
    data_x_bi = tf.layers.conv1d(data_x, filters=150, kernel_size=[
        2], padding="same", activation=tf.nn.elu)
    data_x = tf.concat((data_x_tri, data_x_bi), 2)
    # cell_for = tf.nn.rnn_cell.MultiRNNCell(
    #     [create_lstm_cell(num_units) for num_units in params.lstmUnits])
    # cell_back = tf.nn.rnn_cell.MultiRNNCell(
    #     [create_lstm_cell(num_units) for num_units in params.lstmUnits])
    if len(params.lstmUnits) > 0:
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [create_lstm_cell(num_units) for num_units in params.lstmUnits])
        value, _ = tf.nn.dynamic_rnn(
            cell, data_x, dtype=tf.float32, sequence_length=length(data_x))
        # value = tf.transpose(value, [1, 0, 2])
    else:
        value = data_x
    # ipdb.set_trace()
    # (for_value, back_value), _ = tf.nn.bidirectional_dynamic_rnn(
    #     cell_for, cell_back, data_x, dtype=tf.float32, sequence_length=length(data_x))
    # value = tf.concat([for_value, back_value], axis=2)
    # ipdb.set_trace()
    # dense = tf.gather(value, int(value.get_shape()[0]) - 1)
    dense = tf.layers.flatten(value)

    # dense = tf.layers.batch_normalization(
    #     dense, training=(not predict))
    for layer in params.fc_layer_units:
        dense = tf.layers.dense(inputs=dense, units=layer, activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),)
        dense = tf.nn.dropout(dense, keep_prob=params.dropout_keep_prob)
    # dense = tf.layers.batch_normalization(
    #     dense, training=(not predict))
    logits = tf.layers.dense(
        inputs=dense, units=params.output_classes, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(),)
    predictions = tf.round(logits)
    return logits, predictions


sess = tf.InteractiveSession()
test_val = tf.placeholder(
    tf.float32, shape=(None, max_word_count, 300))
output = build_model(test_val)
saver = tf.train.Saver()
saver.restore(sess, "tensorboard_RNN/model300.ckpt")
data_class = DataClass()


def predict(tweet):
    data_list = tweet.lower().strip("!.?,").split()
    data_list = list(
        filter(lambda x: x not in data_class.stop_words, data_list))

    def padding_function(row):
        # ipdb.set_trace()
        if len(row) < max_word_count:
            row += ["" for i in range(max_word_count - len(row))]
        else:
            row = row[:max_word_count]
        return row
    data_list = padding_function(data_list)
    data_list = data_class.replace_by_word_embeddings(data_list)
    # ipdb.set_trace()
    prediction, _ = sess.run(output, feed_dict={test_val: [data_list]})
    if prediction < 0.4:
        return "Negative"
    elif prediction > 0.4 and prediction < 0.6:
        return "Neutral"
    else:
        return "Positive"
