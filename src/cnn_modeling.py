# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import re
import itertools
import tensorflow as tf
import string
from io import BytesIO
from tensorflow.contrib import learn
from collections import Counter
from time import time
import datetime
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from itertools import izip
import pickle
from utils import notification


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

d = pd.read_csv("data/finals_meta.csv", 
                usecols=('content_split','score_positive','score_hate','score_neutral','hour_tag','n_hate'))

d=d[d['content_split'].notnull()]
d=d[d['n_hate'].notnull()]
d.reset_index(drop=True,inplace=True)

def oneHot(dummy_labels):
    le = preprocessing.LabelEncoder()
    enc = OneHotEncoder()
    
    le.fit (dummy_labels)
    y_dummy = le.fit_transform(dummy_labels)
    y_dummy = y_dummy.reshape(-1, 1)
    enc.fit(y_dummy)
    y_dummy = enc.transform(y_dummy).toarray()
    y_dummy = y_dummy.astype('float32')
    print ("\n * OneHot example")
    print (y_dummy)
    return (y_dummy)


def clean_str(string):
    """
    Tokenization/string cleaning 
    """
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\$", " $ ", string) #yes, isolate $
    string = re.sub(r"\%", " % ", string) #yes, isolate %
    string = re.sub(r"\s{2,}", " ", string)
    
    # fixing XXX and xxx like as word
    string = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx",string)
    
    return string.strip().lower()


word_data=[]
t0 = time()

for message in d['content_split']:
    word_data.append(clean_str(message))

# With a MacBook Pro (Late 2011)
# 2.4 GHz Intel Core i5, 4 GB 1333 MHz DDR3
print ("\nCleaning time: ", round(time()-t0, 1), "s")

max_document_length = max([len(x.split(" ")) for x in word_data])
print ("Max_document_length:",max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
num_data = np.array(list(vocab_processor.fit_transform(word_data)))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
# Check data "lengths"
print ("Check my variables:")
print ("\n* word_data length:", len(word_data))
print ("* num_data length: ", len(num_data)) 

np.random.seed(57)

##  append meta features
num_data_list = num_data.tolist()
for x, y in izip(num_data_list, d['score_positive']):
	x.append(y)

for x, y in izip(num_data_list, d['score_hate']):
	x.append(y)

for x, y in izip(num_data_list, d['score_neutral']):
	x.append(y)

for x, y in izip(num_data_list, oneHot(d['hour_tag'])):
	x.extend(y)


num_data = np.array(num_data_list)

shuffle_indices = np.random.permutation(np.arange(len(num_data)))
x_shuffled = num_data[shuffle_indices]
y_shuffled = d['n_hate'][shuffle_indices]
print ("* x shuffled:", x_shuffled.shape)
print ("* y shuffled:", y_shuffled.shape)

features_dummy, x_test, labels_dummy, test_labels = model_selection.train_test_split(x_shuffled, y_shuffled, test_size=0.20, random_state= 23)
del num_data, d 
x_train, x_valid, train_labels, valid_labels = model_selection.train_test_split(features_dummy, labels_dummy, test_size=0.25, random_state= 34)

print('Training set  ',   x_train.shape, train_labels.shape)
print('Validation set',   x_valid.shape, valid_labels.shape)
print('Test set      ',    x_test.shape,  test_labels.shape)

# free some memory
del x_shuffled, y_shuffled, labels_dummy, features_dummy

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # meta feature
        self.input_score_positive = tf.placeholder(tf.float32, [None, num_classes], name="input_input_score_positive")
        self.input_score_hate = tf.placeholder(tf.float32, [None, num_classes], name="input_score_hate")
        self.input_score_neutral = tf.placeholder(tf.float32, [None, num_classes], name="input_score_neutral")
        self.input_onehot1 = tf.placeholder(tf.float32, [None, num_classes], name="input_onehot1")
        self.input_onehot2 = tf.placeholder(tf.float32, [None, num_classes], name="input_onehot2")
        self.input_onehot3 = tf.placeholder(tf.float32, [None, num_classes], name="input_onehot3")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1], 
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat( pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.h_pool_flat = tf.concat([self.h_pool_flat,self.input_score_positive], 1)
        self.h_pool_flat = tf.concat([self.h_pool_flat,self.input_score_hate], 1)
        self.h_pool_flat = tf.concat([self.h_pool_flat,self.input_score_neutral], 1)
        self.h_pool_flat = tf.concat([self.h_pool_flat,self.input_onehot1], 1)
        self.h_pool_flat = tf.concat([self.h_pool_flat,self.input_onehot2], 1)
        self.h_pool_flat = tf.concat([self.h_pool_flat,self.input_onehot3], 1)

        num_filters_total = num_filters_total+6

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W1 = tf.get_variable(
                "W1",
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            
            self.scores = tf.nn.relu(tf.matmul(self.h_drop, W1) + b1)           
#            self.scores = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="scores") 

            W2 = tf.get_variable(
                "W2",
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b")
            l2_loss += tf.nn.l2_loss(W2)
            l2_loss += tf.nn.l2_loss(b2)

            self.scores = tf.nn.relu(tf.matmul(self.scores, W2) + b2)           
#            self.scores = tf.nn.xw_plus_b(self.scores, W2, b2, name="scores") 
            
            W3 = tf.get_variable(
                "W3",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W3)
            l2_loss += tf.nn.l2_loss(b3)

            self.scores = tf.nn.relu(tf.matmul(self.scores, W3) + b3)           
#            self.scores = tf.nn.xw_plus_b(self.scores, W3, b3, name="scores") 
            
 #           self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            print (self.scores)
            print (self.input_y)
            
            losses = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.input_y, self.scores))))
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '2,3,4')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.05, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 4, "Number of training epochs (best: 8)") # was 200
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

y_train =[]
for i in train_labels:
    y_train.append([i])

y_valid =[]
for i in valid_labels:
    y_valid.append([i])
    
y_test =[]
for i in test_labels:
    y_test.append([i])


# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1]-6,
            num_classes=1,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: tuple([i[:-6] for i in x_batch]),
              cnn.input_score_neutral: tuple([[i[-4]] for i in x_batch]),
              cnn.input_score_hate: tuple([[i[-5]] for i in x_batch]),
              cnn.input_score_positive: tuple([[i[-6]] for i in x_batch]),
              cnn.input_onehot1: tuple([[i[-3]] for i in x_batch]),
              cnn.input_onehot2: tuple([[i[-2]] for i in x_batch]),
              cnn.input_onehot3: tuple([[i[-1]] for i in x_batch]),

              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step , loss = sess.run(
                [train_op, global_step, cnn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: tuple([i[:-6] for i in x_batch]),
              cnn.input_y: y_batch,
              cnn.input_score_neutral: tuple([[i[-1]] for i in x_batch]),
              cnn.input_score_hate: tuple([[i[-2]] for i in x_batch]),
              cnn.input_score_positive: tuple([[i[-3]] for i in x_batch]),
              cnn.input_onehot1: tuple([[i[-3]] for i in x_batch]),
              cnn.input_onehot2: tuple([[i[-2]] for i in x_batch]),
              cnn.input_onehot3: tuple([[i[-1]] for i in x_batch]),

              cnn.dropout_keep_prob: 1.0
            }
            step, loss = sess.run(
                [global_step, cnn.loss],
                feed_dict)
            return loss

        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        
        # Training loop. For each batch...
        for batch in batches:
            qq =  batch
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            
# Validating
# ==================================================
            if current_step % FLAGS.evaluate_every == 0:
                #print("\nEvaluation:")
                
                # Generate batches
                batches_valid = batch_iter(
                    list(zip(x_valid, y_valid)), FLAGS.batch_size, 1)
                
                loss_valid = 0.
                len_batches = 0.
                
                for batch_valid in batches_valid:  
                    
                    x_batch_valid, y_batch_valid = zip(*batch_valid)
                    aLoss = dev_step(x_batch_valid, y_batch_valid)
                    loss_valid += aLoss 
                    len_batches += 1.
                
                loss_valid = loss_valid / len_batches
                time_str = datetime.datetime.now().isoformat()
                print("Validation set: {}, step {}, loss {:g} ".format(time_str, current_step, loss_valid))

    
        
# Testing
# ==================================================
        if True:
            print("\n\nTest set:")
            
            # Generate batches
            batches_test = batch_iter(
                list(zip(x_test, y_test)), FLAGS.batch_size, 1)
        
            loss_test = 0.
            len_batches = 0.
            
            for batch_test in batches_test:  
                    
                    x_batch_test, y_batch_test = zip(*batch_test)
                    #aLoss, anAcc, aSummary = dev_step(x_batch_test, y_batch_test, writer=dev_summary_writer)
                    aLoss = dev_step(x_batch_test, y_batch_test)
                    loss_test += aLoss 
                    len_batches += 1.
                
            loss_test = loss_test / len_batches
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g} ".format(time_str, current_step, loss_test))
            test_log = "{}: step {}, loss {:g} ".format(time_str, current_step, loss_test)
            #dev_summary_writer.add_summary(aSummary, current_step)
            print("")



    notification.notification('cnn model is ready!')

	with open('cnn.pkl', 'wb') as output:
		pickle.dump(cnn, output, pickle.HIGHEST_PROTOCOL)


