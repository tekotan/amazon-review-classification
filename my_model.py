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

max_word_count = 300


class BaseModel(object):
    def __init__(self):
        # add data imports
        self.data_path = "./complete.json.gz"
        self.data_split = [0.9, 0.05, 0.05]
        pass

    def replace_by_word_embeddings(self, row):
        # ipdb.set_trace()
        new_arr = np.zeros((max_word_count, 300))
        for w, word in enumerate(row):
            try:
                new_arr[w] = self.vec_model[word]
            except Exception as e:
                if w == 300:
                    ipdb.set_trace()
                new_arr[w] = np.zeros((300))
        return new_arr.astype(np.float32)

    def get_data_as_df(self, file, total_examples):
        def parse(path):
            g = gzip.open(path, 'rb')
            for l in tqdm(g):
                yield eval(l)

        def getDF(path):
            i = 0
            counter = 0
            df = {}
            for d in tqdm(parse(path)):
                df[i] = d
                i += 1
                if counter > total_examples:
                    break
                else:
                    counter += 1
            data_df = pd.DataFrame.from_dict(
                df, orient='index', dtype=np.float32)
            print(data_df.columns)
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
            print("Loading Vectors")
            # self.vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            #     "vectors/wiki-news-300d-1M.vec")
            print("Completed Loading Vectors")
            self.vec_model = {}

            data_df["reviewText"] = data_df["reviewText"].apply(
                self.replace_by_word_embeddings)
            # for i in range(max_word_count):
            #     ipdb.set_trace()
            #     temp_dict = {}
            #     temp_dict[str(i)] = data_df["reviewText"]
            #     data_df = data_df.assign(**temp_dict)
            # data_df.drop(["reviewText"])

            # ipdb.set_trace()
            # data_df["reviewText"] = pd.to_numeric(data_df["reviewText"])
            # data_df["overall"] = pd.to_numeric(data_df["overall"])

            return data_df
        return getDF(file)

    def create_iterator(self, batch_size):
        data_df = self.get_data_as_df(self.data_path, 10000)

        test_df = data_df[:100]
        train_df = data_df[100:]
        features = []
        for i in train_df["reviewText"]:
            features.append(i.tolist())
        features = np.array(features, dtype=np.float32).reshape(
            (len(train_df), max_word_count, max_word_count))
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (features, train_df["overall"]))
        train_dataset = train_dataset.shuffle(len(train_df))
        train_dataset = train_dataset.batch(batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        features = []
        for i in test_df["reviewText"]:
            features.append(i.tolist())
        features = np.array(features, dtype=np.float32).reshape(
            (len(test_df), max_word_count, max_word_count))

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (features, test_df["overall"]))
        test_dataset = test_dataset.repeat(len(train_df) // len(test_df))
        test_dataset = test_dataset.batch(batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()
        del features
        return train_iterator, test_iterator

    def input_fn(self, params):

        train_iterator, test_iterator = self.create_iterator(params.batch_size)
        (train_x, train_y), (test_x,
                             test_y) = train_iterator.get_next(), test_iterator.get_next()
        sess1 = tf.InteractiveSession()
        sess1.run(tf.global_variables_initializer())
        print(f"\n\n\n\n\n\n\n{sess1.run([train_x, train_y])}\n\n\n\n")
        return (train_x, train_y), (test_x, test_y)


class ConvNetModel(BaseModel):
    def __init__(self, params):
        self.data_path = "./complete.json.gz"
        (train_x, train_y), (test_x, test_y) = self.input_fn(params)
        self.create_model(params, train_x, train_y, test_x, test_y)

    def create_model(self, params, train_x, train_y, test_x, test_y):
        predictions = self.build_model(train_x, params)
        self.loss = tf.reduce_mean(tf.metrics.root_mean_squared_error(labels=train_y,
                                                                      predictions=predictions))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_sum(
            tf.cast(tf.equal(predictions, train_y), tf.float32))
        val_pred = self.build_model(test_x, params)
        self.validation = tf.reduce_sum(
            tf.cast(tf.equal(predictions, test_y), tf.float32))

    def build_model(self, train_x, params):
        # build model
        # pooled_outputs = []
        # for i, filter_size in enumerate(params.filter_sizes):
        #     with tf.name_scope("conv-maxpool-%s" % filter_size):
        #         # Convolution Layer
        #         filter_shape = [filter_size,
        #                         300, 1, params.num_filters]
        #         W = tf.Variable(tf.truncated_normal(
        #             filter_shape, stddev=0.1, dtype=tf.float32), name="W")
        #         b = tf.Variable(tf.constant(
        #             0.1, shape=[params.num_filters], dtype=tf.float32), name="b")
        #         conv = tf.nn.conv2d(
        #             train_x,
        #             W,
        #             strides=[1, 1, 1, 1],
        #             padding="VALID",
        #             name="conv")
        #         # Apply nonlinearity
        #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #         # Max-pooling over the outputs
        #         pooled = tf.nn.max_pool(
        #             h,
        #             ksize=[1, 100 - filter_size + 1, 1, 1],
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name="pool")
        #         pooled_outputs.append(pooled)
        # # Combine all the pooled features
        # num_filters_total = params.num_filters * len(params.filter_sizes)
        # h_pool = tf.concat(pooled_outputs, axis=3)
        # h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        #
        # h_drop = tf.nn.dropout(h_pool_flat, params.dropout_keep_prob)
        #
        # W = tf.Variable(tf.truncated_normal(
        #     [num_filters_total, 1], stddev=0.1, dtype=tf.float32), name="W")
        # b = tf.Variable(tf.constant(
        #     0.1, shape=[1], dtype=tf.float32), name="b")
        # predictions = tf.nn.relu(tf.nn.xw_plus_b(
        #     h_drop, W, b, name="scores
        train_x = tf.reshape(train_x, (-1, 300 * 300))
        W = tf.Variable(tf.truncated_normal(
            [300 * 300, 1], stddev=0.1, dtype=tf.float32), name="W")
        b = tf.Variable(tf.constant(
            0.1, shape=[1], dtype=tf.float32), name="b")
        predictions = tf.nn.relu(tf.nn.xw_plus_b(train_x, W, b))
        return predictions

    def predict(self, predict_x, params):
        predict_x = list(map(self.extract_words, predict_x))
        predict_x = tf.py_func(
            self.lookup, [predict_x], tf.float32, stateful=True, name=None)
        predictions = self.build_model(predict_x, params)
        return sess.run(predictions)


class LSTMModel(BaseModel):
    def __init__(self, params):
        self.data_path = "./complete.json.gz"
        (train_x, train_y), (test_x, test_y) = self.input_fn(params)
        self.create_model(params, train_x, train_y, test_x, test_y)

    def create_model(self, params, train_x, train_y, test_x, test_y):
        predictions = self.model(train_x, params)
        self.loss = tf.reduce_sum(tf.metrics.root_mean_squared_error(labels=train_y,
                                                                     predictions=predictions))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_sum(
            tf.cast(tf.equal(predictions, train_y), tf.float32))
        val_pred = self.model(test_x, params)
        self.validation = tf.reduce_sum(
            tf.cast(tf.equal(predictions, test_y), tf.float32))

    def model(self, train_x, params):
        lstmCell1 = tf.nn.rnn_cell.LSTMCell(params.lstmUnits)
        lstmCell1 = tf.nn.rnn_cell.DropoutWrapper(
            cell=lstmCell1, output_keep_prob=0.9)
        lstmCell2 = tf.nn.rnn_cell.LSTMCell(params.lstmUnits)
        lstmCell2 = tf.nn.rnn_cell.DropoutWrapper(
            cell=lstmCell2, output_keep_prob=0.9)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell1, lstmCell2])
        value, _ = tf.nn.dynamic_rnn(cell, train_x, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal(
            [params.lstmUnits, 1]), name='Weights')
        bias = tf.Variable(tf.constant(0.1, shape=[1]), name='bias')
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)

        predictions = tf.nn.relu(tf.matmul(last, weight) + bias)

        return predictions


class Params(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            exec(f"self.{key} = {val}")


class TrainModel(object):
    def __init__(self):
        pass

    def train(self, Model, params):
        sess = tf.Session()
        train_writer = tf.summary.FileWriter('/train',
                                             sess.graph)
        sess.run(tf.global_variables_initializer())
        print("input_fn")
        print(sess.run(Model.input_fn(params)))
        for iteration in tqdm(range(params.total_iterations)):
            accuracy, validation, _ = sess.run(
                [Model.accuracy, Model.validation, Model.optimizer])
            if iteration % params.report_step:
                with open("accuracy_log.txt", "w") as file:
                    file.write(
                        f"Iteration: {iteration}, Accuracy: {accuracy}, Validation: {validation}")
