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
import logging
import random
import shutil


from gensim.models.wrappers import FastText
# import data_precessing as Data
max_word_count = 20
data_split = [0.8, 0.1, 0.1]


class DataClass(object):
    def __init__(self):
        # add data imports
        self.data_df = pd.read_csv("data.csv")
        self.data_path = "./complete.json.gz"
        self.data_split = data_split
        self.stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
                           'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or',  'who', 'as', 'from', 'him', 'each', 'the', 'themselves',
                           'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their',
                           'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then',
                           'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just',
                           'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                           'further', 'was', 'here', ]
        print("Loading Vectors")
        self.vec_model = FastText.load_fasttext_format(
            'vectors/cc.en.300.bin').wv
        # self.vec_model = {}
        print("Completed Loading Vectors")
        self.data_df["label"] = self.data_df["label"].astype(int)
        self.data_df["text"] = self.data_df["text"].astype(str)
        self.data_df["text"] = self.data_df["text"].str.lower()
        self.data_df["text"] = self.data_df["text"].str.strip(
            to_strip=".!?,")
        self.data_df["text"] = self.data_df["text"].str.split()
        self.data_df["text"] = self.data_df["text"].apply(
            lambda x: [w for w in x if not w in self.stop_words])

    def replace_by_word_embeddings(self, row):
        # ipdb.set_trace()
        new_arr = np.zeros((max_word_count, 300))
        for w, word in enumerate(row):
            try:
                new_arr[w] = self.vec_model[word]
            except Exception as e:
                new_arr[w] = np.zeros((300))
        return new_arr.astype(np.float32)

    def get_amazon_data(self, data_split):
        # def parse(path):
        #     g = gzip.open(path, 'rb')
        #     for l in tqdm(g):
        #         yield eval(l)

        def parse(path):
            g = gzip.open(path, 'rb')
            for l in tqdm(g):
                data_df = pd.DataFrame.from_dict(
                    {0: eval(l)}, orient='index', dtype=np.float32)
                # print(data_df.columns)
                data_df = data_df[["reviewText", "overall"]]
                data_df["reviewText"] = data_df["reviewText"].str.lower()
                data_df["reviewText"] = data_df["reviewText"].str.strip(
                    to_strip=".!?,")
                data_df["reviewText"] = data_df["reviewText"].str.split()

                def padding_function(row):
                    # ipdb.set_trace()
                    if len(row) < max_word_count:
                        row += ["" for i in range(max_word_count - len(row))]
                    else:
                        row = row[:max_word_count]
                    return row
                data_df["reviewText"] = data_df["reviewText"].apply(
                    padding_function)
                # ipdb.set_trace()

                data_df["reviewText"] = data_df["reviewText"].apply(
                    self.replace_by_word_embeddings)
                for i in range(300):
                    # ipdb.set_trace()
                    temp_dict = {}
                    temp_dict[str(i)] = [data_df["reviewText"][0][:, i]]
                    data_df = data_df.assign(**temp_dict)
                # ipdb.set_trace()
                # data_df.drop(["reviewText"])

                # data_df["reviewText"] = pd.to_numeric(data_df["reviewText"])
                data_df["overall"] = pd.to_numeric(data_df["overall"])
                yield data_df

    def get_data_as_df(self, data_split):

        def get_generator(data_split):
            i = 0
            counter = 0
            # df = {}
            # for d in tqdm(parse(path)):
            #     if counter > start_examples:
            #         df[i] = d
            #         i += 1
            #     if counter > stop_examples:
            #         break
            #     else:
            #         counter += 1

            def padding_function(row):
                # ipdb.set_trace()
                if len(row) < max_word_count:
                    row += ["" for i in range(max_word_count - len(row))]
                else:
                    row = row[:max_word_count]
                return row

            def train_data():
                for i in range(data_split[0]):
                    ret_df = self.data_df.iloc[i].copy()
                    ret_df["text"] = padding_function(ret_df["text"])
                    # ipdb.set_trace()

                    ret_df["text"] = self.replace_by_word_embeddings(
                        ret_df["text"])
                    yield (ret_df["text"], [ret_df["label"]])

            def val_data():
                for i in range(data_split[0], data_split[1] + data_split[0]):
                    ret_df = self.data_df.iloc[i].copy()
                    ret_df["text"] = padding_function(ret_df["text"])
                    # ipdb.set_trace()

                    ret_df["text"] = self.replace_by_word_embeddings(
                        ret_df["text"])
                    yield (ret_df["text"], [ret_df["label"]])

            def test_data():
                for i in range(data_split[0] + data_split[1], data_split[2] + data_split[0] + data_split[1]):
                    ret_df = self.data_df.iloc[i].copy()
                    ret_df["text"] = padding_function(ret_df["text"])
                    # ipdb.set_trace()
                    ret_df["text"] = self.replace_by_word_embeddings(
                        ret_df["text"])
                    yield (ret_df["text"], [ret_df["label"]])
            # for i in range(max_word_count):
            #     # ipdb.set_trace()
            #     temp_dict = {}
            #     temp_dict[str(i)] = self.data_df["text"][:, i]
            #     self.data_df = self.data_df.assign(**temp_dict)

            # self.data_df.drop(["text"])

            # data_df["text"] = pd.to_numeric(data_df["text"])
            # data_df["label"] = pd.to_numeric(data_df["label"])
            # ipdb.set_trace()
            return train_data, val_data, test_data
        return get_generator(data_split)


class BaseModel(object):
    def __init__(self):
        # add data imports
        self.data_path = "./complete.json.gz"
        self.data_split = data_split
        self.data_class = DataClass()

    def input_fn(self, batch_size, epochs):
        # split = list(map(lambda perc: int(
        #     round(Data.TOTAL_EXAMPLES * perc)), self.data_split))
        #
        # def extract_fn(data_record):
        #     feature_dict = {}
        #     for i in range(300):
        #         feature_dict[str(i)] = tf.FixedLenFeature(
        #             [max_word_count], tf.float32)
        #     feature_dict["overall"] = tf.FixedLenFeature([1], tf.float32)
        #     sample = tf.parse_single_example(data_record, feature_dict)
        #     return sample
        # self.train_dataset = tf.data.TFRecordDataset(
        #     [f"train_data/train_{i}.tfrecord" for i in range(0, split[0])])
        # self.train_dataset = self.train_dataset.shuffle(split[0] // 2)
        # self.train_dataset = self.train_dataset.map(extract_fn)
        # self.train_dataset = self.train_dataset.repeat(epochs * 3)
        # self.train_dataset = self.train_dataset.batch(batch_size)
        #
        # self.val_dataset = tf.data.TFRecordDataset(
        #     [f"val_data/val_{i}.tfrecord" for i in range(0, split[1] - 1)])
        # # self.val_dataset = self.val_dataset.shuffle(split[1] // 30)
        # self.val_dataset = self.val_dataset.map(extract_fn)
        # self.val_dataset = self.val_dataset.repeat(
        #     (split[0] // split[1]) * epochs * batch_size)
        # self.val_dataset = self.val_dataset.batch(batch_size)
        #
        # self.val_dataset_total = tf.data.TFRecordDataset(
        #     [f"val_data/val_{i}.tfrecord" for i in range(0, split[1] - 1)])
        # self.val_dataset_total = self.val_dataset_total.map(extract_fn)
        # self.val_dataset_total = self.val_dataset_total.repeat(
        #     (split[0] // split[1]) * epochs * batch_size)
        # self.val_dataset_total = self.val_dataset_total.batch(
        #     len(os.listdir("val_data/")))
        #
        # self.test_dataset = tf.data.TFRecordDataset(
        #     [f"test_data/test_{i}.tfrecord" for i in range(split[2] - 1)])
        # self.test_dataset = self.test_dataset.map(extract_fn)
        # self.test_dataset = self.test_dataset.shuffle(split[2])
        # self.test_dataset = self.test_dataset.repeat(
        #     (split[0] // split[2]) * epochs * batch_size)
        # self.test_dataset = self.test_dataset.batch(5)
        #
        # self.handle = tf.placeholder(tf.string, shape=[])
        #
        # self.iterator = tf.data.Iterator.from_string_handle(
        #     self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
        #
        # data = self.iterator.get_next()
        # data_Y = data["overall"]
        # data.pop("overall")
        # data_X = list(data.values())
        #
        # return data_X, tf.reshape(tf.one_hot(tf.cast(data_Y, tf.int32), 6), (-1, 6))
        split = list(map(lambda perc: int(
            round(1600000 * perc)), self.data_split))

        self.train, self.val, self.test = self.data_class.get_data_as_df(split)
        self.train_dataset = tf.data.Dataset.from_generator(
            self.train, (tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
        self.train_dataset = self.train_dataset.shuffle(1000)
        self.train_dataset = self.train_dataset.repeat()
        self.train_dataset = self.train_dataset.batch(batch_size)

        self.val_dataset = tf.data.Dataset.from_generator(
            self.val, (tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
        # self.val_dataset = self.val_dataset.shuffle(split[1] // 30)
        self.val_dataset = self.val_dataset.repeat()
        self.val_dataset = self.val_dataset.batch(batch_size)

        self.val_dataset_total = tf.data.Dataset.from_generator(
            self.val, (tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
        self.val_dataset_total = self.val_dataset_total.batch(split[1] - 1)

        self.test_dataset = tf.data.Dataset.from_generator(
            self.test, (tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
        self.test_dataset = self.test_dataset.shuffle(1000)
        self.test_dataset = self.test_dataset.repeat()
        self.test_dataset = self.test_dataset.batch(1)

        self.handle = tf.placeholder(tf.string, shape=[])

        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)

        data_x, data_y = self.iterator.get_next()

        return data_x, tf.reshape(tf.map_fn(lambda x: x / 4, data_y), (-1, 1))


class ConvNetModel(BaseModel):
    def __init__(self, params, predict=False):
        self.data_path = "./complete.json.gz"
        self.data_split = data_split
        self.type = "CNN"

        self.data_x, self.data_y = self.input_fn(
            params.batch_size, params.num_epochs)
        # ipdb.set_trace()
        if not predict:
            self.create_model(params, self.data_x, self.data_y)

    def create_model(self, params, data_x, data_y):
        self.build_model(data_x, params, False)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=data_y,
                                                                              logits=self.logits))
        tf.summary.scalar("loss", self.loss)
        self._optimizer = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = self._optimizer.minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(data_y, 1), self.predictions), tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def build_model(self, data_x, params, predict=False):
        # build model
        data_x = tf.reshape(data_x, (-1, max_word_count, 300))
        pooled_outputs = []
        for i, filter_size in enumerate(params.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size,
                                300, 1, params.num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1, dtype=tf.float32), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[params.num_filters], dtype=tf.float32), name="b")
                data_x = tf.reshape(data_x, (-1, max_word_count, 300, 1))
                conv = tf.nn.conv2d(
                    data_x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 100 - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = params.num_filters * len(params.filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis=3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total * 201])

        h_drop = tf.nn.dropout(h_pool_flat, params.dropout_keep_prob)
        # ipdb.set_trace()
        W = tf.Variable(tf.truncated_normal(
            [num_filters_total * 201, 6], stddev=0.1, dtype=tf.float32), name="W")
        b = tf.Variable(tf.constant(
            0.1, shape=[6], dtype=tf.float32), name="b")
        self.predictions = tf.nn.softmax(tf.nn.xw_plus_b(
            h_drop, W, b, name="scores"))
        if predict:
            return self.predictions
        # data_x = tf.reshape(data_x, (-1, 300 * 300))
        # self.W = tf.Variable(tf.truncated_normal(
        #     [300 * 300, 1], stddev=0.1, dtype=tf.float32), name="W")
        # self.b = tf.Variable(tf.constant(
        #     0.1, shape=[1], dtype=tf.float32), name="b")
        # self.predictions = tf.nn.relu(tf.nn.xw_plus_b(data_x, self.W, self.b))

    def build_model(self, data_x, params, predict=False):
        # build model
        data_x = tf.reshape(data_x, (-1, 1, max_word_count, 300))

        conv = tf.layers.conv2d(data_x, filters=64, kernel_size=[
            20, 20], padding="same", activation=tf.nn.relu)
        # conv = tf.layers.batch_normalization(conv, training=(not predict))
        # conv = tf.layers.conv2d(conv, filters=64, kernel_size=[
        #                         5, 5], padding="same", activation=tf.nn.relu)
        # conv = tf.layers.batch_normalization(conv, training=(not predict))
        # conv = tf.layers.max_pooling2d(
        #     conv, pool_size=[2, 2], strides=2, padding="same")

        # conv = tf.layers.conv2d(conv, filters=128, kernel_size=[
        #                         2, 2], padding="same", activation=tf.nn.relu)
        # conv = tf.layers.batch_normalization(conv, training=(not predict))
        # conv = tf.layers.conv2d(conv, filters=128, kernel_size=[
        #                         3, 3], padding="same", activation=tf.nn.relu)
        # conv = tf.layers.batch_normalization(conv, training=(not predict))
        # conv = tf.layers.max_pooling2d(
        #     conv, pool_size=[2, 2], strides=2, padding="same")

        # conv = tf.layers.conv2d(conv, filters=256, kernel_size=[
        #                         2, 2], padding="same", activation=tf.nn.relu)
        # conv = tf.layers.batch_normalization(conv, training=(not predict))
        # conv = tf.layers.conv2d(conv, filters=256, kernel_size=[
        #                         3, 3], padding="same", activation=tf.nn.relu)
        # conv = tf.layers.batch_normalization(conv, training=(not predict))
        # conv = tf.layers.max_pooling2d(
        #     conv, pool_size=[2, 2], strides=2, padding="same")

        # conv = tf.layers.conv2d(conv, filters=512, kernel_size=[
        #                         2, 2], padding="same", activation=tf.nn.relu)
        # conv = tf.layers.batch_normalization(conv, training=(not predict))
        # conv = tf.layers.conv2d(conv, filters=512, kernel_size=[
        #                         3, 3], padding="same", activation=tf.nn.relu)
        # conv = tf.layers.batch_normalization(conv, training=(not predict))
        # conv = tf.layers.max_pooling2d(
        #     conv, pool_size=[2, 2], strides=2, padding="same")

        dense = tf.layers.flatten(conv)

        for layer in params.fc_layer_units:
            dense = tf.layers.dense(inputs=dense, units=layer, activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),)
            dense = tf.nn.dropout(dense, keep_prob=params.dropout_keep_prob)
        # dense = tf.layers.batch_normalization(
        #     dense, training=(not predict))
        self.logits = tf.layers.dense(
            inputs=dense, units=params.output_classes, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),)

        # weight = tf.Variable(tf.truncated_normal(
        #     [params.fc_layer_units[-1], params.output_classes]), name='Weights')
        # bias = tf.Variable(tf.constant(
        #     0.1, shape=[params.output_classes]), name='bias')
        # self.logits = tf.matmul(dense, weight) + bias
        self.predictions = tf.argmax(self.logits, axis=1)
        if predict:
            return self.logits


class LSTMModel(BaseModel):
    def __init__(self, params, predict=False):
        self.data_path = "./complete.json.gz"
        self.data_split = data_split
        self.type = "RNN"
        if not predict:
            self.data_class = DataClass()
            self.data_x, self.data_y = self.input_fn(
                params.batch_size, params.num_epochs)
            self.create_model(params, self.data_x, self.data_y)

    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def create_model(self, params, data_x, data_y):
        self.build_model(data_x, params, False)
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_y,
        #                                                                    logits=self.logits))
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=data_y, predictions=self.logits))
        tf.summary.scalar("loss", self.loss)
        self._optimizer = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = self._optimizer.minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            data_y, self.predictions), tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def build_model(self, data_x, params, predict):
        # ipdb.set_trace()
        data_x = tf.reshape(data_x, (-1, max_word_count, 300))
        # data_x = tf.layers.batch_normalization(data_x, training=(not predict))

        def create_lstm_cell(units):
            lstmCell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                units, activation=tf.nn.sigmoid, dropout_keep_prob=params.dropout_keep_prob)
            # lstmCell = tf.nn.rnn_cell.LSTMCell(
            #     units, activation=tf.nn.sigmoid)
            # lstmCell = tf.contrib.rnn.AttentionCellWrapper(lstmCell, 5)
            return lstmCell
        # cell_for = tf.nn.rnn_cell.MultiRNNCell(
        #     [create_lstm_cell(num_units) for num_units in params.lstmUnits])
        # cell_back = tf.nn.rnn_cell.MultiRNNCell(
        #     [create_lstm_cell(num_units) for num_units in params.lstmUnits])
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [create_lstm_cell(num_units) for num_units in params.lstmUnits])

        value, _ = tf.nn.dynamic_rnn(
            cell, data_x, dtype=tf.float32, sequence_length=self.length(data_x))
        # (for_value, back_value), _ = tf.nn.bidirectional_dynamic_rnn(
        #     cell_for, cell_back, data_x, dtype=tf.float32, sequence_length=self.length(data_x))
        # value = tf.concat([for_value, back_value], axis=2)
        value = tf.transpose(value, [1, 0, 2])
        dense = tf.gather(value, int(value.get_shape()[0]) - 1)
        # dense = tf.layers.batch_normalization(
        #     dense, training=(not predict))
        for layer in params.fc_layer_units:
            dense = tf.layers.dense(inputs=dense, units=layer, activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),)
            dense = tf.nn.dropout(dense, keep_prob=params.dropout_keep_prob)
        dense = tf.layers.batch_normalization(
            dense, training=(not predict))
        self.logits = tf.layers.dense(
            inputs=dense, units=params.output_classes, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(),)

        # weight = tf.Variable(tf.truncated_normal(
        #     [params.fc_layer_units[-1], params.output_classes]), name='Weights')
        # bias = tf.Variable(tf.constant(
        #     0.1, shape=[params.output_classes]), name='bias')
        # self.logits = tf.matmul(dense, weight) + bias
        self.predictions = tf.round(self.logits)
        if predict:
            return self.logits


class Params(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if not hasattr(self, key):
                # exec(f"self.{key} = {val}")
                setattr(self, key, val)


class RunModel(object):
    def __init__(self, Model, params, predict=False):
        self.Model = Model
        self.params = params
        if predict:
            print("Loading Vectors")
            self.vec_model = FastText.load_fasttext_format(
                'vectors/cc.en.300.bin/cc.en.300.bin').wv
            print("Finished Loading Vectors")
            # self.vec_model = []
            print("loading model")
            self.test_val = tf.placeholder(
                tf.float32, shape=(None, max_word_count, 300))
            self.test_output = self.Model.build_model(
                self.test_val, self.params, predict=True)
            self.sess = tf.InteractiveSession()
            self.saver = tf.train.Saver()
            # self.sess.run(tf.global_variables_initializer())
            # self.sess.run(tf.local_variables_initializer())
            self.saver.restore(self.sess, tf.train.latest_checkpoint(
                f"tensorboard_{self.Model.type}"))
            print("finished Loading model")

    def predict_during_train(self, data_x):
        data_list = data_x.lower().strip("!.?,").split()

        def padding_function(row):
            # ipdb.set_trace()
            if len(row) < max_word_count:
                row += ["" for i in range(max_word_count - len(row))]
            else:
                row = row[:max_word_count]
            return row
        # ipdb.set_trace()
        data_list = padding_function(data_list)
        data_list = self.replace_by_word_embeddings(data_list)
        data_list = data_list.reshape((1, 300, 20))
        data_dict = {"overall": 5.0}
        for i in range(300):
            data_dict[str(i)] = data_list[0][i]
        predict_dataset = tf.data.Dataset.from_tensor_slices(
            data_dict).batch(1)

        return predict_dataset

    def predict_during_train_iter(self, sess):
        print("Loading vectors")
        # self.vec_model = FastText.load_fasttext_format(
        #     'vectors/cc.en.300.bin/cc.en.300.bin').wv
        print("Finished Loading Vectors")
        # ipdb.set_trace()
        pred_iter1 = self.predict_during_train(
            "great").make_one_shot_iterator()
        pred_iter2 = self.predict_during_train(
            "bad").make_one_shot_iterator()

        pred_handle1 = sess.run(pred_iter1.string_handle())
        pred_handle2 = sess.run(pred_iter2.string_handle())

        predict_1 = sess.run(self.Model.logits, feed_dict={
            self.Model.handle: pred_handle1})
        predict_2 = sess.run(self.Model.logits, feed_dict={
            self.Model.handle: pred_handle2})
        print(predict_1, predict_2)

    def get_sent(self, vectors):
        word_list = []
        for i in vectors:
            if (i != np.zeros([300])).all():
                word_list.append(
                    self.Model.data_class.vec_model.similar_by_vector(i)[0][0])
        return " ".join(word_list)

    def train(self, restore_ckpt):
        logging.basicConfig(
            filename=f'train_{self.Model.type}.log', level=logging.DEBUG)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # ipdb.set_trace()
            if restore_ckpt:
                if os.path.isdir(f"tensorboard_{self.Model.type}") and tf.train.latest_checkpoint(f"tensorboard_{self.Model.type}"):
                    saver.restore(sess, tf.train.latest_checkpoint(
                        f"tensorboard_{self.Model.type}"))
            else:
                shutil.rmtree(
                    f"tensorboard_{self.Model.type}", ignore_errors=True)
            train_writer = tf.summary.FileWriter(
                f"tensorboard_{self.Model.type}", sess.graph)
            train_iterator = self.Model.train_dataset.make_one_shot_iterator()
            val_iterator = self.Model.val_dataset.make_one_shot_iterator()
            val_iterator_total = self.Model.val_dataset_total.make_one_shot_iterator()

            test_iterator = self.Model.test_dataset.make_one_shot_iterator()

            train_handle = sess.run(train_iterator.string_handle())
            val_handle = sess.run(val_iterator.string_handle())
            val_handle_total = sess.run(val_iterator_total.string_handle())
            test_handle = sess.run(test_iterator.string_handle())
            ipdb.set_trace()
            for epoch in range(self.params.num_epochs):
                # range(len(os.listdir("train_data/")) // self.params.batch_size + 1)):
                for iteration in tqdm(range(1600000 // (self.params.batch_size + 1))):
                    try:
                        # ipdb.set_trace()
                        accuracy, loss, _ = sess.run(
                            [self.Model.accuracy, self.Model.loss, self.Model.optimizer], feed_dict={self.Model.handle: train_handle})
                        validation_acc, validation_loss = sess.run([self.Model.accuracy, self.Model.loss], feed_dict={
                            self.Model.handle: val_handle})
                        if iteration % self.params.report_step == 0:
                            summary = sess.run(self.Model.merged, feed_dict={
                                self.Model.handle: train_handle})
                            train_writer.add_summary(
                                summary, iteration * epoch + 1)
                            print_val_pred = ", ".join(sess.run(self.Model.logits, feed_dict={
                                self.Model.handle: test_handle}).reshape(1).astype(str))
                            print_val_truth = self.get_sent(sess.run(self.Model.data_x, feed_dict={
                                self.Model.handle: test_handle})[0])
                            logging.info(
                                f"Iteration: {iteration*(epoch+1)}, Loss: {loss}, Accuracy: {accuracy}, Validation Accuracy: {validation_acc}, Validation Loss: {validation_loss}, Test Sentence: {print_val_truth} Test Ground Prediction: {print_val_pred}")
                        if (iteration) % self.params.save_step == 0 and iteration > 0:
                            save_path = saver.save(
                                sess, f"./tensorboard_{self.Model.type}/model{iteration*(epoch+1)}.ckpt")
                            logging.info(
                                f"Saved Checkpoint at iteration {iteration*(epoch+1)} and path {save_path}")
                    except tf.errors.OutOfRangeError:
                        saver.save(
                            sess, f"./tensorboard_{self.Model.type}/model{iteration*(epoch+1)}.ckpt")
                        sys.exit()
                # ipdb.set_trace()
                logging.info(
                    "################################################################################################")
                logging.info(f"Finished Epoch {epoch}")
                validation_acc, validation_loss = sess.run([self.Model.accuracy, self.Model.loss], feed_dict={
                    self.Model.handle: val_handle_total})
                logging.info(
                    f"Validation Accuracy: {validation_acc}, Validation Loss: {validation_loss}")
            # self.predict_during_train_iter(sess)
            save_path = saver.save(
                sess, f"./tensorboard_{self.Model.type}/model{iteration*epoch}.ckpt")
            logging.info(
                f"Saved Checkpoint at iteration {iteration*epoch} and path {save_path}")

    def replace_by_word_embeddings(self, row):
        # ipdb.set_trace()
        new_arr = np.zeros((max_word_count, 300))
        for w, word in enumerate(row):
            try:
                new_arr[w] = self.vec_model[word]
            except Exception as e:
                new_arr[w] = np.zeros((300))
        return new_arr.astype(np.float32)

    def predict(self, data_x):
        data_list = data_x.lower().strip("!.?,").split()

        def padding_function(row):
            # ipdb.set_trace()
            if len(row) < max_word_count:
                row += ["" for i in range(max_word_count - len(row))]
            else:
                row = row[:max_word_count]
            return row
        data_list = padding_function(data_list)
        data_list = self.replace_by_word_embeddings(data_list)
        # ipdb.set_trace()
        return self.sess.run(self.test_output, feed_dict={self.test_val: [data_list]})
