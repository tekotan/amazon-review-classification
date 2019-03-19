from model_params import params
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
from my_model import DataClass

from gensim.models.wrappers import FastText
import data_precessing as Data
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


if train:
    batch_size = params.batch_size
    predict = False
    restore_ckpt = False

    split = list(map(lambda perc: int(
        round(1600000 * perc)), data_split))
    data_class = DataClass()
    train, val, test = data_class.get_data_as_df(split)
    train_dataset = tf.data.Dataset.from_generator(
        train, (tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.from_generator(
        val, (tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
    # val_dataset = val_dataset.shuffle(split[1] // 30)
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(batch_size)

    val_dataset_total = tf.data.Dataset.from_generator(
        val, (tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
    val_dataset_total = val_dataset_total.batch(split[1] - 1)

    test_dataset = tf.data.Dataset.from_generator(
        test, (tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
    test_dataset = test_dataset.shuffle(1000)
    test_dataset = test_dataset.repeat()
    test_dataset = test_dataset.batch(1)

    handle = tf.placeholder(tf.string, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    data_x, data_y = iterator.get_next()

    data_one_hot = tf.reshape(tf.map_fn(lambda x: x / 4, data_y), (-1, 1))

    data_x = tf.reshape(data_x, (-1, max_word_count, 300))
    logits, predictions = build_model(data_x)
    type = "RNN"
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=data_one_hot, logits=logits))
    tf.summary.scalar("loss", loss)
    _optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    optimizer = _optimizer.minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        data_one_hot, predictions), tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # ipdb.set_trace()
        if restore_ckpt:
            if os.path.isdir(f"tensorboard_{type}") and tf.train.latest_checkpoint(f"tensorboard_{type}"):
                saver.restore(sess, tf.train.latest_checkpoint(
                    f"tensorboard_{type}"))
        else:
            shutil.rmtree(
                f"tensorboard_{type}", ignore_errors=True)
        train_writer = tf.summary.FileWriter(
            f"tensorboard_{type}", sess.graph)
        train_iterator = train_dataset.make_one_shot_iterator()
        val_iterator = val_dataset.make_one_shot_iterator()
        val_iterator_total = val_dataset_total.make_one_shot_iterator()

        test_iterator = test_dataset.make_one_shot_iterator()

        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())
        val_handle_total = sess.run(val_iterator_total.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        # ipdb.set_trace()
        for epoch in range(params.num_epochs):
            # range(len(os.listdir("train_data/")) // params.batch_size + 1)):
            for iteration in range(1600000 // (params.batch_size + 1)):
                try:
                    # ipdb.set_trace()
                    _accuracy, _loss, _ = sess.run(
                        [accuracy, loss, optimizer], feed_dict={handle: train_handle})
                    validation_acc, validation_loss = sess.run([accuracy, loss], feed_dict={
                        handle: val_handle})
                    if iteration % params.report_step == 0:
                        summary = sess.run(merged, feed_dict={
                            handle: train_handle})
                        train_writer.add_summary(
                            summary, iteration * epoch + 1)
                        print_val_pred = ", ".join(sess.run(logits, feed_dict={
                            handle: test_handle}).reshape(1).astype(str))
                        # print_val_truth = get_sent(sess.run(data_x, feed_dict={
                        #     handle: test_handle})[0])
                        ground_truth = sess.run(
                            data_one_hot, feed_dict={handle: test_handle})[0]
                        tf.logging.info(
                            f"Iteration: {iteration*(epoch+1)}, Loss: {_loss}, Accuracy: {_accuracy}, Validation Accuracy: {validation_acc}, Validation Loss: {validation_loss}, Test Ground Prediction: {print_val_pred}, {ground_truth}")
                    if (iteration) % params.save_step == 0 and iteration > 0:
                        save_path = saver.save(
                            sess, f"./tensorboard_{type}/model{iteration*(epoch+1)}.ckpt")
                        tf.logging.info(
                            f"Saved Checkpoint at iteration {iteration*(epoch+1)} and path {save_path}")
                except tf.errors.OutOfRangeError:
                    saver.save(
                        sess, f"./tensorboard_{type}/model{iteration*(epoch+1)}.ckpt")
                    sys.exit()
            # ipdb.set_trace()
            tf.logging.info(
                "################################################################################################")
            tf.logging.info(f"Finished Epoch {epoch}")
            validation_acc, validation_loss = sess.run([accuracy, loss], feed_dict={
                handle: val_handle_total})
            tf.logging.info(
                f"Validation Accuracy: {validation_acc}, Validation Loss: {validation_loss}")
        # predict_during_train_iter(sess)
        save_path = saver.save(
            sess, f"./tensorboard_{type}/model{iteration*epoch}.ckpt")
        tf.logging.info(
            f"Saved Checkpoint at iteration {iteration*epoch} and path {save_path}")
else:
    with tf.Session() as sess:
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
            return sess.run(output, feed_dict={test_val: [data_list]})
        ipdb.set_trace()
