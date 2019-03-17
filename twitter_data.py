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
import shutil
from sklearn.utils import shuffle

global train_num
global test_num
global val_num

train_num = 0
val_num = 0
test_num = 0

train = 1000000
test = 300000
val = 299999
at_a_time = 10000


def build_feature_dict(data):
    feature_dict = {}
    # ipdb.set_trace()
    try:
        embedding_array = data["text"][0]
    except:
        try:
            embedding_array = data["text"][:, 1]
        except:
            embedding_array = np.array(data["text"].tolist()[0])
    for vec in range(300):
        feature_dict[str(vec)] = tf.train.Feature(
            float_list=tf.train.FloatList(value=embedding_array[:, vec]))
    feature_dict["labels"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[data["label"]]))
    return feature_dict


def main():
    global train_num
    global test_num
    global val_num
    from my_model import DataClass
    from my_model import max_word_count
    dirNames = ["twitter_train_data", "twitter_val_data", "twitter_test_data"]
    for dirName in dirNames:
        try:
            os.mkdir(dirName)
        except FileExistsError:
            pass
    data_class = DataClass()
    with tf.python_io.TFRecordWriter(f'twitter_train_data/train.tfrecord') as writer:
        for index in tqdm(range(0, train - 1)):
            data_df = data_class.get_data_as_df(index, index + 1)
            train_data = tf.train.Example(features=tf.train.Features(
                feature=build_feature_dict(data_df)))
            writer.write(train_data.SerializeToString())
            train_num += 1
    with tf.python_io.TFRecordWriter(f'twitter_test_data/test.tfrecord') as writer:
        for index in tqdm(range(train, test - 1)):
            data_df = data_class.get_data_as_df(index, index + 1)
            train_data = tf.train.Example(features=tf.train.Features(
                feature=build_feature_dict(data_df)))
            writer.write(train_data.SerializeToString())
            test_num += 1
    with tf.python_io.TFRecordWriter(f'twitter_val_data/val.tfrecord') as writer:
        for index in tqdm(range(test + train, val - 1)):
            data_df = data_class.get_data_as_df(index, index + 1)
            train_data = tf.train.Example(features=tf.train.Features(
                feature=build_feature_dict(data_df)))
            writer.write(train_data.SerializeToString())
            val_num += 1


if __name__ == "__main__":
    main()
