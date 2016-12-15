#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import tensorflow as tf
import numpy as np
import sys, os, shutil

# data = np.array([1, 2, 3, 4, 5])
# print data[1:1+10]
# exit()

SEQ_LENGTH = 21
VOCA_SIZE = 100
BATCH_SIZE = 256
RNN_SIZE = 256
LOG_DIR_PATH = "./logs/rnn"

if os.path.exists(LOG_DIR_PATH):
    shutil.rmtree(LOG_DIR_PATH)
os.makedirs(LOG_DIR_PATH)

NUM_BATCHES = 200
data = np.random.randint(VOCA_SIZE, size=(BATCH_SIZE * NUM_BATCHES, SEQ_LENGTH))

data_idx = 0


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def get_batch():
    global data_idx
    start = (data_idx % (NUM_BATCHES - 1)) * BATCH_SIZE
    # print "data_idx %d, start %d(%d)\n" \ % (data_idx, start, start*BATCH_SIZE)

    input = data[start:start + BATCH_SIZE, :]
    target = (2 * input + 1) % VOCA_SIZE

    data_idx = data_idx + 1

    return input.astype(float), target

def build_rnn(labels):
    with tf.variable_scope("RNN"):
        embedding = weight_variable([VOCA_SIZE, RNN_SIZE], name="embedding/weight")

        lstm = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE)

        # Initial state of the LSTM memory.
        #state = tf.zeros([BATCH_SIZE, RNN_SIZE])
        lstm_h = tf.zeros(shape=[BATCH_SIZE, RNN_SIZE], name="lstm_h")
        lstm_c = tf.zeros(shape=[BATCH_SIZE, RNN_SIZE], name="lstm_c")
        state = [lstm_h, lstm_c]

        # FC
        fc1_w = weight_variable([RNN_SIZE, VOCA_SIZE], name="embedding/weight")
        fc1_b = bias_variable([VOCA_SIZE], name="embedding/weight")

        logits = []
        for seq_idx in range(SEQ_LENGTH):
            if seq_idx > 0:
                tf.get_variable_scope().reuse_variables()
            # The value of state is updated after processing each batch of words.
            output, state = lstm(tf.nn.embedding_lookup(embedding, labels[:, seq_idx]), state)

            # The LSTM output can be used to make next word predictions

            logits.append(tf.matmul(output, fc1_w) + fc1_b)

        return logits

def rnn_loss(logits, target):
    with tf.variable_scope("LOSS"):
        loss = 0
        for seq_idx in range(SEQ_LENGTH):
            #if seq_idx > 0:
            #    tf.get_variable_scope().reuse_variables()
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits[seq_idx], target[:, seq_idx])
            loss += tf.reduce_mean(cross_entropy)
        return loss

def build_reg(labels):
    with tf.variable_scope("RNN"):
        labels = tf.reshape(labels, shape=[-1, 1])
        fc1_w = weight_variable([1, 50], name="fc1/weight")
        fc1_b = bias_variable([50], name="fc1/b")
        fc1_h = tf.nn.relu(tf.matmul(labels, fc1_w) + fc1_b)

        fc2_w = weight_variable([50, 50], name="fc2/weight")
        fc2_b = bias_variable([50], name="fc2/b")
        fc2_h = tf.nn.relu(tf.matmul(fc1_h, fc2_w) + fc2_b)

        fc3_w = weight_variable([50, VOCA_SIZE], name="fc3/weight")
        fc3_b = bias_variable([VOCA_SIZE], name="fc3/b")
        fc3_h = tf.nn.relu(tf.matmul(fc2_h, fc3_w) + fc3_b)

        return fc3_h

def reg_loss(logits, target):
    with tf.variable_scope("LOSS"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        return cross_entropy_mean

def train_rnn():
    input = tf.placeholder(shape=[BATCH_SIZE, SEQ_LENGTH], dtype=tf.int64)
    target = tf.placeholder(shape=[BATCH_SIZE, SEQ_LENGTH], dtype=tf.int64)

    logits = build_rnn(input)
    loss = rnn_loss(logits, target)
    tf.scalar_summary("rnn/loss", loss)

    learning_rate = tf.placeholder(tf.float32)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    predicts = tf.arg_max(tf.transpose(logits, perm=[1, 0, 2]), 2)

    accuracy = tf.mul(tf.reduce_sum(tf.cast(tf.equal(predicts, target), tf.float32)), 100 / (SEQ_LENGTH * BATCH_SIZE))
    tf.scalar_summary("rnn/accuracy", accuracy)

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter(LOG_DIR_PATH, sess.graph)
        merged = tf.merge_all_summaries()

        tf.initialize_all_variables().run()

        best_accuracy = 0.0
        min_loss = 10000.0
        LEARNING_RATE = 0.01

        g_idx = 0

        while True:
            input_value, target_value = get_batch()
            _, summary, output_value, loss_value, accuracy_value \
                = sess.run([train_op, merged, predicts, loss, accuracy],
                           feed_dict={
                               learning_rate: LEARNING_RATE,
                               input: input_value,
                               target: target_value
                           })

            if best_accuracy < accuracy_value:
                best_accuracy = accuracy_value

            if min_loss > loss_value:
                min_loss = loss_value

            if g_idx % 100 == 0:
                print "iter %03d, lr : %0.5f\nloss : %0.2f(min %0.2f), accuracy : %0.2f(best %0.2f)" \
                      % (g_idx, LEARNING_RATE, loss_value, min_loss, accuracy_value, best_accuracy)

                for batch_idx in range(min(5, BATCH_SIZE)):
                    for seq_idx in range(min(5, SEQ_LENGTH)):
                        sys.stdout.write("%3d" % input_value[batch_idx][seq_idx])
                    sys.stdout.write("  ==>")
                    for seq_idx in range(min(5, SEQ_LENGTH)):
                        sys.stdout.write("%3d" % output_value[batch_idx][seq_idx])
                    sys.stdout.write("  ==>")
                    for seq_idx in range(min(5, SEQ_LENGTH)):
                        sys.stdout.write("%3d" % target_value[batch_idx][seq_idx])
                    sys.stdout.write("\n")

                # save log
                writer.add_summary(summary, g_idx)

            g_idx = g_idx + 1

            if g_idx != 0 and g_idx % 1000 == 0:
                LEARNING_RATE = LEARNING_RATE * 0.5


def train_reg():
    input = tf.placeholder(shape=[BATCH_SIZE, SEQ_LENGTH], dtype=tf.float32)
    target = tf.placeholder(shape=[BATCH_SIZE, SEQ_LENGTH], dtype=tf.int64)

    input_slice = input[:, 0]
    target_slice = target[:, 0]

    logits = build_reg(input_slice)
    loss = reg_loss(logits, target_slice)
    tf.scalar_summary("rnn/loss", loss)

    learning_rate = tf.placeholder(tf.float32)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    predicts = tf.arg_max(logits, 1)

    accuracy = tf.reduce_sum(tf.cast(tf.equal(predicts, target_slice), tf.float32)) * 100 / BATCH_SIZE
    tf.scalar_summary("rnn/accuracy", accuracy)

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter(LOG_DIR_PATH, sess.graph)
        merged = tf.merge_all_summaries()

        tf.initialize_all_variables().run()

        best_accuracy = 0.0
        min_loss = 10000.0
        LEARNING_RATE = 0.01

        g_idx = 0

        while True:
            input_value, target_value = get_batch()

            _, output_value, loss_value, accuracy_value \
                = sess.run([train_op, predicts, loss, accuracy],
                           feed_dict={
                               learning_rate: LEARNING_RATE,
                               input: input_value,
                               target: target_value
                           })

            if best_accuracy < accuracy_value:
                best_accuracy = accuracy_value


            if min_loss > loss_value:
                min_loss = loss_value

            if g_idx % 1000 == 0:
                _, summary, output_value, loss_value, accuracy_value \
                    = sess.run([train_op, merged, predicts, loss, accuracy],
                               feed_dict={
                                   learning_rate: LEARNING_RATE,
                                   input: input_value,
                                   target: target_value
                               })

                print "iter %03d, lr : %0.5f\nloss : %0.2f(min %0.2f), accuracy : %0.2f(best %0.2f)" \
                      % (g_idx, LEARNING_RATE, loss_value, min_loss, accuracy_value, best_accuracy)

                for batch_idx in range(min(5, BATCH_SIZE)):
                    sys.stdout.write("%3d" % input_value[batch_idx][0])
                    sys.stdout.write("  ==>")
                    sys.stdout.write("%3d" % output_value[batch_idx])
                    sys.stdout.write("  ==>")
                    sys.stdout.write("%3d" % target_value[batch_idx][0])
                    sys.stdout.write("\n")

                # save log
                writer.add_summary(summary, g_idx)


            g_idx = g_idx + 1

            if g_idx != 0 and g_idx % 10000 == 0:
                LEARNING_RATE = LEARNING_RATE * 0.99

train_rnn()
#train_reg()