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

TOTAL_EXAMPLES = 100000
at_a_time = 1000
train_num = 0
val_num = 0
test_num = 0


def main():
    from my_model import DataClass
    dirNames = ["train_data", "val_data", "test_data"]
    for dirName in dirNames:
        try:
            os.mkdir(dirName)
        except FileExistsError:
            shutil.rmtree(dirName)
            os.mkdir(dirName)
    data_class = DataClass()

    def data_process(start, stop, iter):
        global train_num
        global test_num
        global val_num

        data_as_df = data_class.get_data_as_df(
            data_class.data_path, start, stop)

        data_split = list(map(lambda perc: int(
            round(at_a_time * perc)), data_class.data_split))

        def build_feature_dict(data):
            feature_dict = {}
            # ipdb.set_trace()
            for vec in range(len(data["reviewText"])):
                feature_dict[str(vec)] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.array(data["reviewText"])[vec, :]))
            feature_dict["overall"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[data["overall"]]))
            return feature_dict

        for i in range(0, data_split[0]):
            # ipdb.set_trace()
            train_data = tf.train.Example(features=tf.train.Features(
                feature=build_feature_dict(data_as_df.iloc[i])))

            with tf.python_io.TFRecordWriter(f'train_data/train_{train_num}.tfrecord') as writer:
                writer.write(train_data.SerializeToString())
            train_num += 1
        # ipdb.set_trace()
        for x in range(data_split[0], data_split[0] + data_split[1]):
            # ipdb.set_trace()
            val_data = tf.train.Example(features=tf.train.Features(
                feature=build_feature_dict(data_as_df.iloc[x])))
            # ipdb.set_trace()
            with tf.python_io.TFRecordWriter(f'val_data/val_{val_num}.tfrecord') as writer:
                writer.write(val_data.SerializeToString())
            val_num += 1
        for y in range(data_split[0] + data_split[1], data_split[0] + data_split[1] + data_split[2]):
            # ipdb.set_trace()
            test_data = tf.train.Example(features=tf.train.Features(
                feature=build_feature_dict(data_as_df.iloc[y])))

            with tf.python_io.TFRecordWriter(f'test_data/test_{test_num}.tfrecord') as writer:
                writer.write(test_data.SerializeToString())
            test_num += 1

    for i in range(0, TOTAL_EXAMPLES, at_a_time):
        data_process(i, i + at_a_time, i // at_a_time)


if __name__ == "__main__":
    main()
