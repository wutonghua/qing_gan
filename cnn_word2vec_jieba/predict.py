#!/usr/bin/python
# -*- coding: utf-8 -*-
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import word2vec_helpers
from text_cnn import TextCNN
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("input_text_file", "./data/spam_100.utf8", "Test text data source to evaluate.")
tf.flags.DEFINE_string("input_label_file", "", "Label file for test text data source.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1522808531/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# validate
# ==================================================

# validate checkout point file
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print("Using checkpoint file : {}".format(checkpoint_file))

# validate word2vec model file
trained_word2vec_model_file = os.path.join(FLAGS.checkpoint_dir,"trained_word2vec.model")
if not os.path.exists(trained_word2vec_model_file):
    print("Word2vec model file \'{}\' doesn't exist!".format(trained_word2vec_model_file))
print("Using word2vec model file : {}".format(trained_word2vec_model_file))

# validate training params file
training_params_file = os.path.join(FLAGS.checkpoint_dir,"training_params.pickle")
if not os.path.exists(training_params_file):
    print("Training params file \'{}\' is missing!".format(training_params_file))
print("Using training params file : {}".format(training_params_file))

# Load params
params = data_helpers.loadDict(training_params_file)
num_labels = int(params['num_labels'])
max_document_length = int(params['max_document_length'])




# Evaluation
# ==================================================
# print("\nEvaluating...\n")
# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
# graph = tf.Graph()
# with graph.as_default():
#     session_conf = tf.ConfigProto(
#       allow_soft_placement=FLAGS.allow_soft_placement,
#       log_device_placement=FLAGS.log_device_placement)
#     sess = tf.Session(config=session_conf)
#     with sess.as_default():
#         # Load the saved meta graph and restore variables
#         saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#         saver.restore(sess, checkpoint_file)
#
#         # Get the placeholders from the graph by name
#         input_x = graph.get_operation_by_name("input_x").outputs[0]
#         # input_y = graph.get_operation_by_name("input_y").outputs[0]
#         dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
#
#         # Tensors we want to evaluate
#         predictions = graph.get_operation_by_name("output/predictions").outputs[0]
#         while True:
#             string=input('请输入:')
#             x_raw =[data_helpers.clean_str(data_helpers.seperate_line(string))]
#             # Get Embedding vector x_test
#             sentences, max_document_length = data_helpers.padding_sentences(x_raw, '<PADDING>',
#                                                                             padding_sentence_length=max_document_length)
#             x_test = np.array(word2vec_helpers.embedding_sentences(sentences, file_to_load=trained_word2vec_model_file))
#             # print("x_test.shape = {}".format(x_test.shape))
#             pred = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})[0]
#             if pred==0:
#                 lei_bie='neg'
#             else:
#                 lei_bie='pos'
#             print(lei_bie)

# Evaluation
# ==================================================
print("\nPredicting...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
sess = tf.Session()
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
saver.restore(sess, checkpoint_file)
graph = tf.get_default_graph()
# Get the placeholders from the graph by name
input_x = graph.get_operation_by_name("input_x").outputs[0]
# input_y = graph.get_operation_by_name("input_y").outputs[0]
dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

# Tensors we want to evaluate
predictions = graph.get_operation_by_name("output/predictions").outputs[0]
def pre_type(string,max_document_length):
    x_raw = [data_helpers.clean_str(data_helpers.jieba_line(string))]
    # Get Embedding vector x_test
    sentences, max_document_length = data_helpers.padding_sentences(x_raw, '<PADDING>',
                                                                    padding_sentence_length=max_document_length)
    x_test = np.array(word2vec_helpers.embedding_sentences(sentences, file_to_load=trained_word2vec_model_file))
    # print("x_test.shape = {}".format(x_test.shape))
    pred = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})[0]
    type=''
    if pred == 0:
        type = type + 'neg'
    else:
        type = type + 'pos'
    return type

if __name__=='__main__':
    while True:
        string = input('请输入:')
        type=pre_type(string,max_document_length)
        print(type)




