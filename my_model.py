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


from gensim.models.wrappers import FastText
import data_precessing as Data
max_word_count = 40
data_split = [0.8, 0.1, 0.1]


class DataClass(object):
    def __init__(self):
        # add data imports
        self.data_path = "./complete.json.gz"
        self.data_split = data_split
        print("Loading Vectors")
        self.vec_model = FastText.load_fasttext_format(
            'vectors/cc.en.300.bin/cc.en.300.bin').wv
        # self.vec_model = {}
        print("Completed Loading Vectors")

    def replace_by_word_embeddings(self, row):
        # ipdb.set_trace()
        new_arr = np.zeros((max_word_count, 300))
        for w, word in enumerate(row):
            try:
                new_arr[w] = self.vec_model[word]
            except Exception as e:
                new_arr[w] = np.zeros((300))
        return new_arr.astype(np.float32)

    def get_data_as_df(self, file):  # , start_examples, stop_examples):
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

        # def getDF(path):
        #     i = 0
        #     counter = 0
        #     # df = {}
        #     # for d in tqdm(parse(path)):
        #     #     if counter > start_examples:
        #     #         df[i] = d
        #     #         i += 1
        #     #     if counter > stop_examples:
        #     #         break
        #     #     else:
        #     #         counter += 1
        #     gen_data = parse(path)
        #     data_df = pd.DataFrame.from_dict(
        #         {0: next(gen_data)}, orient='index', dtype=np.float32)
        #     # print(data_df.columns)
        #     data_df = data_df[["reviewText", "overall"]]
        #     data_df["reviewText"] = data_df["reviewText"].str.lower()
        #     data_df["reviewText"] = data_df["reviewText"].str.strip(
        #         to_strip=".!?,")
        #     data_df["reviewText"] = data_df["reviewText"].str.split()
        #
        #     def padding_function(row):
        #         # ipdb.set_trace()
        #         if len(row) < max_word_count:
        #             row += ["" for i in range(max_word_count - len(row))]
        #         else:
        #             row = row[:max_word_count]
        #         return row
        #     data_df["reviewText"] = data_df["reviewText"].apply(
        #         padding_function)
        #     # ipdb.set_trace()
        #
        #     data_df["reviewText"] = data_df["reviewText"].apply(
        #         self.replace_by_word_embeddings)
        #     for i in range(max_word_count):
        #         ipdb.set_trace()
        #         temp_dict = {}
        #         temp_dict[str(i)] = data_df["reviewText"]
        #         data_df = data_df.assign(**temp_dict)
        #     data_df.drop(["reviewText"])
        #
        #     ipdb.set_trace()
        #     data_df["reviewText"] = pd.to_numeric(data_df["reviewText"])
        #     data_df["overall"] = pd.to_numeric(data_df["overall"])
        #
        #     return data_df
        return parse(file)


class BaseModel(object):
    def __init__(self):
        # add data imports
        self.data_path = "./complete.json.gz"
        self.data_split = data_split

    def input_fn(self, batch_size, epochs):
        split = list(map(lambda perc: int(
            round(Data.TOTAL_EXAMPLES * perc)), self.data_split))

        def extract_fn(data_record):
            feature_dict = {}
            for i in range(300):
                feature_dict[str(i)] = tf.FixedLenFeature(
                    [max_word_count], tf.float32)
            feature_dict["overall"] = tf.FixedLenFeature([1], tf.float32)
            sample = tf.parse_single_example(data_record, feature_dict)
            return sample
        self.train_dataset = tf.data.TFRecordDataset(
            [f"train_data/train_{i}.tfrecord" for i in range(0, split[0])])
        self.train_dataset = self.train_dataset.shuffle(split[0] // 30)
        self.train_dataset = self.train_dataset.map(extract_fn)
        self.train_dataset = self.train_dataset.repeat(epochs * batch_size)
        self.train_dataset = self.train_dataset.batch(batch_size)

        self.val_dataset = tf.data.TFRecordDataset(
            [f"val_data/val_{i}.tfrecord" for i in range(0, split[1] - 1)])
        # self.val_dataset = self.val_dataset.shuffle(split[1] // 30)
        self.val_dataset = self.val_dataset.map(extract_fn)
        self.val_dataset = self.val_dataset.repeat(
            (split[0] // split[1]) * epochs * batch_size)
        self.val_dataset = self.val_dataset.batch(batch_size)

        self.val_dataset_total = tf.data.TFRecordDataset(
            [f"val_data/val_{i}.tfrecord" for i in range(0, split[1] - 1)])
        self.val_dataset_total = self.val_dataset_total.map(extract_fn)
        self.val_dataset_total = self.val_dataset_total.repeat(
            (split[0] // split[1]) * epochs * batch_size)
        self.val_dataset_total = self.val_dataset_total.batch(
            len(os.listdir("val_data/")))

        self.test_dataset = tf.data.TFRecordDataset(
            [f"test_data/test_{i}.tfrecord" for i in range(split[2] - 1)])
        self.test_dataset = self.test_dataset.map(extract_fn)
        self.val_dataset = self.val_dataset.repeat(
            (split[0] // split[2]) * epochs * batch_size)
        self.test_dataset = self.test_dataset.batch(5)

        self.handle = tf.placeholder(tf.string, shape=[])

        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)

        data = self.iterator.get_next()
        data_Y = data["overall"]
        data.pop("overall")
        data_X = list(data.values())

        return data_X, tf.reshape(tf.one_hot(tf.cast(data_Y, tf.int32), 6), (-1, 6))


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
        self.build_model(data_x, params)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=data_y,
                                                               logits=self.predictions)
        tf.summary.scalar("loss", tf.reduce_sum(self.loss))
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params.learning_rate).minimize(self.loss)
        self.total_accuracy, self.accuracy = tf.metrics.accuracy(
            tf.argmax(data_y, 1), tf.argmax(self.predictions, 1))
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


class LSTMModel(BaseModel):
    def __init__(self, params, predict=False):
        self.data_path = "./complete.json.gz"
        self.data_split = data_split
        self.type = "RNN"
        if not predict:
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

    def build_model(self, data_x, params, predict):
        # ipdb.set_trace()
        data_x = tf.reshape(data_x, (-1, max_word_count, 300))
        data_x = tf.layers.batch_normalization(data_x, training=(not predict))

        def create_lstm_cell(units):
            lstmCell = tf.nn.rnn_cell.LSTMCell(
                units, initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.leaky_relu)
            lstmCell = tf.nn.rnn_cell.DropoutWrapper(
                cell=lstmCell, output_keep_prob=params.dropout_keep_prob)
            return lstmCell
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [create_lstm_cell(num_units) for num_units in params.lstmUnits])
        value, _ = tf.nn.dynamic_rnn(
            cell, data_x, dtype=tf.float32, sequence_length=self.length(data_x))

        value = tf.transpose(value, [1, 0, 2])
        dense = tf.gather(value, int(value.get_shape()[0]) - 1)
        for layer in params.fc_layer_units:
            dense = tf.layers.dense(inputs=dense, units=layer, activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),)

        weight = tf.Variable(tf.truncated_normal(
            [params.fc_layer_units[-1], params.output_classes]), name='Weights')
        bias = tf.Variable(tf.constant(
            0.1, shape=[params.output_classes]), name='bias')
        self.logits = tf.matmul(dense, weight) + bias
        self.predictions = tf.argmax(self.logits, axis=1)
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
            self.vec_model = FastText.load_fasttext_format(
                'vectors/cc.en.300.bin/cc.en.300.bin').wv
            # self.vec_model = []
            self.test_val = tf.placeholder(
                tf.float32, shape=(None, max_word_count, 300))
            self.test_output = self.Model.build_model(
                self.test_val, self.params, predict=True)
            self.sess = tf.InteractiveSession()
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, tf.train.latest_checkpoint(
                f"tensorboard_{self.Model.type}"))
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

    def train(self):
        logging.basicConfig(
            filename=f'train_{self.Model.type}.log', level=logging.DEBUG)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            ipdb.set_trace()
            if os.path.isdir(f"tensorboard_{self.Model.type}") and tf.train.latest_checkpoint(f"tensorboard_{self.Model.type}"):
                saver.restore(sess, tf.train.latest_checkpoint(
                    f"tensorboard_{self.Model.type}"))
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
            for epoch in range(self.params.num_epochs):
                for iteration in tqdm(range(len(os.listdir("train_data/")) // self.params.batch_size + 1)):
                    try:
                        accuracy, loss, _ = sess.run(
                            [self.Model.accuracy, self.Model.loss, self.Model.optimizer], feed_dict={self.Model.handle: train_handle})
                        validation_acc, validation_loss = sess.run([self.Model.accuracy, self.Model.loss], feed_dict={
                            self.Model.handle: val_handle})
                        if iteration % self.params.report_step == 0:
                            summary = sess.run(self.Model.merged, feed_dict={
                                self.Model.handle: train_handle})
                            train_writer.add_summary(
                                summary, iteration * epoch)
                            print_val = sess.run(self.Model.predictions, feed_dict={
                                self.Model.handle: test_handle})
                            logging.info(
                                f"Iteration: {iteration*(epoch+1)}, Loss: {loss}, Accuracy: {accuracy}, Validation Accuracy: {validation_acc}, Validation Loss: {validation_loss}, Test: {print_val}")
                        if iteration % self.params.save_step == 0 and iteration > 0:
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
                if w == 300:
                    ipdb.set_trace()
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
