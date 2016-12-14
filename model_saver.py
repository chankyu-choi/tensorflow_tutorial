#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import tensorflow as tf

import os

tf.set_random_seed(0)

DATA_SIZE = 10
BATCH_SIZE = 512


CKPT_DIR_PATH = "./ckpt_dir"

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def build_model(input):
    with tf.name_scope("FC1"):
        fc1_w = weight_variable(shape=[DATA_SIZE, DATA_SIZE], name="fc1/weight")
        fc1_b = bias_variable([DATA_SIZE], name="fc1/bias")
        fc1_h = tf.matmul(input, fc1_w) + fc1_b
        fc1_h = tf.nn.relu(fc1_h)
    with tf.name_scope("FC2"):
        fc2_w = weight_variable(shape=[DATA_SIZE, 2], name="fc2/weight")
        fc2_b = bias_variable([2], name="fc2/bias")
        fc2_h = tf.matmul(fc1_h, fc2_w) + fc2_b
    return fc2_h

def get_batch():
    indices = tf.random_shuffle(tf.to_int32(tf.linspace(0.0, BATCH_SIZE-1, BATCH_SIZE)))
    input = tf.concat(0, [tf.random_normal(shape=[int(BATCH_SIZE/2), DATA_SIZE], mean=0.0, stddev=1.0),
                     tf.random_normal(shape=[int(BATCH_SIZE/2), DATA_SIZE], mean=10.0, stddev=1.0)])
    input = tf.gather(input, indices)
    target = tf.concat(0, [tf.zeros(shape=[int(BATCH_SIZE/2)]), tf.ones(shape=[int(BATCH_SIZE/2)])])
    target = tf.gather(target, indices)
    return input, tf.to_int64(target)

def train():
    LEARNING_RATE = 0.05

    if os.path.exists(CKPT_DIR_PATH) == False:
        os.makedirs(CKPT_DIR_PATH)

    global_step = tf.Variable(0, dtype=tf.int32, name="global_step", trainable=False)
    learning_rate = tf.placeholder(dtype=tf.float32, name="learning_late")

    input, target = get_batch()
    logits = build_model(input)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    predict = tf.arg_max(logits, 1)
    accuracy = tf.reduce_sum(tf.to_float(tf.equal(predict, target)))*100/BATCH_SIZE

    fc1_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "FC1")
    print "fc1_var_list"
    print fc1_var_list

    fc2_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "FC2")
    print "fc2_var_list"
    print fc2_var_list

    var_list = fc1_var_list
    saver = tf.train.Saver(var_list=var_list)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        ckpt = tf.train.get_checkpoint_state(CKPT_DIR_PATH)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables
            saver = tf.train.Saver()

        g_idx = global_step.eval()

        while True:
            _, loss_value, input_value, target_value, predict_value, accuracy_value \
                = sess.run([train_op, loss, input, target, predict, accuracy], feed_dict={
                learning_rate:LEARNING_RATE
            })
            print "iter : %d, loss : %0.3f, accuracy : %0.1f, lr : %0.5f" % (g_idx, loss_value, accuracy_value, LEARNING_RATE)
            for batch_idx in range(5):
                print "%0.2f\t %0.2f\t %0.2f\t => output : %d, target : %d" \
                          % (input_value[batch_idx][0], input_value[batch_idx][0], input_value[batch_idx][0], \
                        predict_value[batch_idx], target_value[batch_idx])

            global_step.assign(g_idx).eval()
            saver.save(sess, CKPT_DIR_PATH + "/model.ckpt", global_step=global_step)

            g_idx = g_idx + 1
            LEARNING_RATE = LEARNING_RATE * 0.999

            if accuracy_value > 99:
                break

def valid():
    global_step = tf.Variable(0, dtype=tf.int32, name="global_step", trainable=False)
    input, target = get_batch()
    logits = build_model(input)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target))

    predict = tf.arg_max(logits, 1)
    accuracy = tf.reduce_sum(tf.to_float(tf.equal(predict, target)))*100/BATCH_SIZE

    fc1_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "FC1")
    print "fc1_var_list"
    print fc1_var_list

    fc2_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "FC2")
    print "fc2_var_list"
    print fc2_var_list

    #var_list = fc1_var_list + fc2_var_list
    var_list = fc1_var_list
    #saver = tf.train.Saver(var_list=var_list)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        if True:
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR_PATH)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables

        g_idx = global_step.eval()


        loss_value, input_value, target_value, predict_value, accuracy_value \
            = sess.run([loss, input, target, predict, accuracy])
        print "iter : %d, loss : %0.3f, accuracy : %0.1f" % (g_idx, loss_value, accuracy_value)
        for batch_idx in range(5):
            print "%0.2f\t %0.2f\t %0.2f\t => output : %d, target : %d" \
                  % (input_value[batch_idx][0], input_value[batch_idx][0], input_value[batch_idx][0], \
                     predict_value[batch_idx], target_value[batch_idx])



#train()
valid()
