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

from my_model import BaseModel

dirNames = ["train_data", "val_data", "test_data"]
for dirName in dirNames:
    try:
        os.mkdir(dirName)
    except FileExistsError:
        shutil.rmtree(dirName)
        os.mkdir(dirName)


TOTAL_EXAMPLES = 1000000
data_class = BaseModel()

data_as_df = data_class.get_data_as_df(data_class.data_path, TOTAL_EXAMPLES)

data_split = list(map(lambda perc: int(
    round(TOTAL_EXAMPLES * perc)), data_class.data_split))


def build_feature_dict(data):
    feature_dict = {}
    for vec in range(len(data["reviewText"])):
        feature_dict[str(vec)] = tf.train.Feature(
            float_list=tf.train.FloatList(value=np.array(data["reviewText"])[:, vec]))
    feature_dict["overall"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[data["overall"]]))
    return feature_dict


for i in range(0, data_split[0]):
    # ipdb.set_trace()
    train_data = tf.train.Example(features=tf.train.Features(
        feature=build_feature_dict(data_as_df.iloc[i])))

    with tf.python_io.TFRecordWriter(f'train_data/train{i}.tfrecord') as writer:
        writer.write(train_data.SerializeToString())

for i in range(data_split[0], data_split[1]):
    # ipdb.set_trace()
    val_data = tf.train.Example(features=tf.train.Features(
        feature=build_feature_dict(data_as_df.iloc[i])))

    with tf.python_io.TFRecordWriter(f'val_data/val{i}.tfrecord') as writer:
        writer.write(val_data.SerializeToString())

for i in range(data_split[1], data_split[2]):
    # ipdb.set_trace()
    test_data = tf.train.Example(features=tf.train.Features(
        feature=build_feature_dict(data_as_df.iloc[i])))

    with tf.python_io.TFRecordWriter(f'test_data/test{i}.tfrecord') as writer:
        writer.write(test_data.SerializeToString())
