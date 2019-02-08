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

TOTAL_EXAMPLES = 20000
train_num = 0
val_num = 0
test_num = 0
global num_ratings_dict
num_ratings_dict = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0
}


def main():
    from my_model import DataClass
    from my_model import max_word_count
    dirNames = ["train_data", "val_data", "test_data"]
    for dirName in dirNames:
        try:
            os.mkdir(dirName)
        except FileExistsError:
            shutil.rmtree(dirName)
            os.mkdir(dirName)
    data_class = DataClass()

    def data_process():
        global train_num
        global test_num
        global val_num
        data_as_gen = data_class.get_data_as_df(
            data_class.data_path)

        data_split = list(map(lambda perc: int(
            round(TOTAL_EXAMPLES * perc)), data_class.data_split))

        def calculate_eligibility(data):
            for value in num_ratings_dict.values():
                if num_ratings_dict[str(int(data["overall"]))] - value > 100:
                    return False
                else:
                    return True

        def build_feature_dict(data):
            feature_dict = {}
            # ipdb.set_trace()
            for vec in range(300):
                feature_dict[str(vec)] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=data[str(vec)][0]))
            feature_dict["overall"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[data["overall"]]))
            return feature_dict
        index = 0
        i = 0
        # ipdb.set_trace()
        while i < data_split[0]:
            # ipdb.set_trace()
            data_as_df = next(data_as_gen)
            if calculate_eligibility(data_as_df):
                train_data = tf.train.Example(features=tf.train.Features(
                    feature=build_feature_dict(data_as_df)))

                with tf.python_io.TFRecordWriter(f'train_data/train_{train_num}.tfrecord') as writer:
                    writer.write(train_data.SerializeToString())
                train_num += 1
                num_ratings_dict[str(
                    int(data_as_df["overall"]))] += 1
                i += 1
            # ipdb.set_trace()
        x = data_split[0] + 1
        # ipdb.set_trace()
        while x > data_split[0] and x < (data_split[0] + data_split[1]):
            # ipdb.set_trace()
            data_as_df = next(data_as_gen)
            if calculate_eligibility(data_as_df):
                train_data = tf.train.Example(features=tf.train.Features(
                    feature=build_feature_dict(data_as_df)))

                with tf.python_io.TFRecordWriter(f'val_data/val_{val_num}.tfrecord') as writer:
                    writer.write(train_data.SerializeToString())
                val_num += 1
                num_ratings_dict[str(
                    int(data_as_df["overall"]))] += 1
                x += 1
        y = data_split[0] + data_split[1] + 1
        # ipdb.set_trace()
        while y > (data_split[0] + data_split[1]) and y < (data_split[0] + data_split[1] + data_split[2]):
            # ipdb.set_trace()
            data_as_df = next(data_as_gen)
            if calculate_eligibility(data_as_df):
                train_data = tf.train.Example(features=tf.train.Features(
                    feature=build_feature_dict(data_as_df)))

                with tf.python_io.TFRecordWriter(f'test_data/test_{test_num}.tfrecord') as writer:
                    writer.write(train_data.SerializeToString())
                test_num += 1
                num_ratings_dict[str(
                    int(data_as_df["overall"]))] += 1
                y += 1
        print("##################################################")
        print(num_ratings_dict)
    data_process()


if __name__ == "__main__":
    main()
