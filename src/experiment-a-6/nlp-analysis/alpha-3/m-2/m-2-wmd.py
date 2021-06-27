#!/usr/bin/env python
# coding: utf-8

'''
To run this script execute the following command:
python m-2-wmd.py /path/to/mb-def-model.h5 /path/to/paired/data

/path/to/mb-def-model.h5 is the path to CNN model trained on MockingBird defended dataset
/path/to/paired/data is the path to the paired dataset generated from the test data
'''

import os
import sys
import string
import random
import re
import itertools

import tensorflow as tf
tf.keras.backend.clear_session()
import pandas as pd
import numpy as np

from tensorflow.keras import optimizers
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, ELU
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical, plot_model

# TF-IDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) != 3:
    print('please provide appropriate number of arguments.')
    sys.exit(-1)

# path to CNN model
# CNN_PATH = '../webfp-crossed-fingerprints/closed-world/models/wt-def-250/wt-def-keyword-100-classes-2000-250-direction-model.h5'
CNN_PATH = sys.argv[1]
# path to the dataset
# data_dir = '../webfp-crossed-fingerprints-data/closed-world/wt-def-keyword/wt-analysis-three-traces-250/paired-data/wt-def-keyword-2-paired.csv'
data_dir = sys.argv[2]

# path to NLP model
NLP_MODEL_PATH = '../../../../data/nlp/models/GoogleNews-vectors-negative300.bin.gz'
# path of text mapping
text_map_path = '../../../../data/nlp/text-mapping/text-map.csv'

nb_classes = 100
input_shape = (2000, 1)


# ### Building CNN model

params = {
        "optimizer": "Adamax",
        "epochs": 110,
        "learning_rate": 0.0033718063908218135,
        "act_fn_1": "relu",
        "act_fn_2": "relu",
        "act_fn_3": "softsign",
        "act_fn_4": "tanh",
        "filter_nums_1": 16,
        "filter_nums_2": 128,
        "filter_nums_3": 128,
        "filter_nums_4": 32,
        "kernel_size": 11,
        "conv_stride_size": 1,
        "pool_stride_size": 5,
        "pool_size": 5,
        "drop_rate": 0.2,
        "drop_rate_fc1": 0.1,
        "drop_rate_fc2": 0.4,
        "fc1_neuron": 256,
        "fc_1_act": "selu",
        "fc2_neuron": 1024,
        "fc_2_act": "selu",
        "batch_size": 32,
        "features": 2000
    }


optimizer = optimizers.Adamax(lr=params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-09, decay = 0.0)


def build_model(nb_classes):
    model = Sequential()
    filter_num = ['None', params['filter_nums_1'], params['filter_nums_2'], params['filter_nums_3'],
                  params['filter_nums_4']]
    kernel_size = ['None', params['kernel_size'], params['kernel_size'], params['kernel_size'],
                   params['kernel_size']]
    conv_stride_size = ['None', params['conv_stride_size'], params['conv_stride_size'], params['conv_stride_size'],
                        params['conv_stride_size']]
    pool_stride_size = ['None', params['pool_stride_size'], params['pool_stride_size'], params['pool_stride_size'],
                        params['pool_stride_size']]
    pool_size = ['None', params['pool_size'], params['pool_size'], params['pool_size'], params['pool_size']]

    # block 1
    model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=(params['features'], 1),
                     strides=conv_stride_size[1], padding='same',
                     name='block1_conv1'))
    model.add(BatchNormalization(axis=-1))
    model.add(ELU(alpha=1.0, name='block1_adv_act1'))
    model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                     strides=conv_stride_size[1], padding='same',
                     name='block1_conv2'))
    model.add(BatchNormalization(axis=-1))
    model.add(ELU(alpha=1.0, name='block1_adv_act2'))
    model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                           padding='same', name='block1_pool'))
    model.add(Dropout(rate=params['drop_rate'], name='block1_dropout'))

    # block 2
    model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                     strides=conv_stride_size[2], padding='same',
                     name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation(params["act_fn_1"], name='block2_act1'))

    model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                     strides=conv_stride_size[2], padding='same',
                     name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation(params["act_fn_1"], name='block2_act2'))
    model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                           padding='same', name='block2_pool'))
    model.add(Dropout(rate=params['drop_rate'], name='block2_dropout'))

    # block 3
    model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                     strides=conv_stride_size[3], padding='same',
                     name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation(params["act_fn_2"], name='block3_act1'))
    model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                     strides=conv_stride_size[3], padding='same',
                     name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation(params["act_fn_2"], name='block3_act2'))
    model.add(MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                           padding='same', name='block3_pool'))
    model.add(Dropout(rate=params['drop_rate'], name='block3_dropout'))

    # block 4
    model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                     strides=conv_stride_size[4], padding='same',
                     name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation(params["act_fn_3"], name='block4_act1'))
    model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                     strides=conv_stride_size[4], padding='same',
                     name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation(params["act_fn_3"], name='block4_act2'))
    model.add(MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                           padding='same', name='block4_pool'))
    model.add(Dropout(rate=params['drop_rate'], name='block4_dropout'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(params['fc1_neuron'], kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation(params["fc_1_act"], name='fc1_act'))
    model.add(Dropout(rate=params['drop_rate_fc1'], name='fc1_dropout'))

    model.add(Dense(params['fc2_neuron'], kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation(params["fc_2_act"], name='fc2_act'))
    model.add(Dropout(rate=params['drop_rate_fc2'], name='fc2_dropout'))

    model.add(Dense(nb_classes, kernel_initializer=glorot_uniform(seed=0), name='fc5'))
    model.add(Activation("softmax", name="pred_layer"))
    
    return model


print('Building and compiling a model')
model = build_model(nb_classes)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
print('Model built and compiled successfully!')
print('loading model weights ...')
model.load_weights(CNN_PATH)
print('weights loaded successfully!')


# Defining function for loading data
def load_test_data(features): 
    """
    This method loads the dataset which is not defended.
    """
    # Point to the directory storing data
    dataset_dir = data_dir

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load testing data
    all_data = pd.read_csv(dataset_dir)
    print('Testing data loaded successfully.')
    
    # Spearating features and output
    x_test = all_data.iloc[:, 1:features+1]
    y_test = all_data.pop('label')

    nb_classes = len(np.unique(y_test))
    print('number of classes in dataset: ', nb_classes)

    print('Data dimensions:')
    print('X testing data shape: ', x_test.shape)
    print('y testing data shape: ', y_test.shape)

    return x_test, y_test


# Reading the data
x_test_temp, y_test_temp = load_test_data(params["features"])


x_test_temp = x_test_temp.to_numpy()
x_test_temp = x_test_temp[:, :,np.newaxis]
y_test_temp = to_categorical(y_test_temp, nb_classes)


score, accuracy = model.evaluate(x_test_temp, y_test_temp, verbose=0)
print('testing score {}, and accuracy {}'.format(score, accuracy))


# ### NLP model and additional libraries

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer
nltk.download('wordnet')
sb = SnowballStemmer("english")

lem = nltk.stem.wordnet.WordNetLemmatizer()

print('loading nlp model ...')
from gensim.models import KeyedVectors
word2vec = KeyedVectors.load_word2vec_format(NLP_MODEL_PATH, binary = True)
# Normalizaing so all the vectors are of same length
word2vec.init_sims(replace=True)
print('nlp model loaded successfully')


# #### Reading the testing data

# Reading the data
x_test, y_test = load_test_data(params["features"])

nb_labels = len(np.unique(y_test))
print('Number of unique labels are: ', nb_labels)


y_test = y_test.astype(int)
# creating one dataframe containig testing data
frames = [x_test, y_test]
testing_data = pd.concat(frames, axis=1, sort=False)
testing_data.head()


# testing_data.shape
# read csv file containing mapping of text files and label
text_map = pd.read_csv(text_map_path)
text_map.head()


# converting data to numpy
x_test = x_test.to_numpy()
x_test = x_test[:, :,np.newaxis]

y_test = to_categorical(y_test, nb_classes)


def get_similarity(l1, l2):
    """
    This function computes text similarity among the predicted labels.
    Word Movers distance is used for computing semantic distance between the text.
    
    l1: label 1
    l2: label 2
    return: the distance between the pair of traces
    """
    web_1_temp = text_map.loc[text_map['encoded-label'] == l1]
    text_1_path = web_1_temp.iloc[0][2]
    with open(text_1_path, 'r') as f1:
        text_1 = f1.read()
        text_1 = re.sub(r'[^\w\s]', '', str(text_1).lower().strip())
        lst_text = text_1.split()
        
        if stop_words is not None:
            lst_text = [word for word in lst_text if word not in stop_words]
        
        lst_text = [sb.stem(word) for word in lst_text]
        lst_text = [lem.lemmatize(word) for word in lst_text]
        text_1 = lst_text
#         text_1 = " ".join(lst_text)

    web_2_temp = text_map.loc[text_map['encoded-label'] == l2]
    text_2_path = web_2_temp.iloc[0][2]
    with open(text_2_path, 'r') as f2:
        text_2 = f2.read()
        text_2 = re.sub(r'[^\w\s]', '', str(text_2).lower().strip())
        lst_text = text_2.split()
        
        if stop_words is not None:
            lst_text = [word for word in lst_text if word not in stop_words]
        
        lst_text = [sb.stem(word) for word in lst_text]
        lst_text = [lem.lemmatize(word) for word in lst_text]
        text_2 = lst_text
        #         text_2 = " ".join(lst_text)
              
    sim = round(word2vec.wmdistance(text_1, text_2), 4)
#     print('Similarity between', l1, 'and', l2, 'is', sim)

    return sim


def get_top_n(trace_preds, top_n):
    '''
    This method gets top-n predictions for the label.
    :param model: the cnn model for the prediction
    :param trace_preds: the predictions of the traffic trace
    :return: The top-n predictions for a particular trace
    '''
    for k in range(len(trace_preds)):
        prob_vec = sorted(trace_preds[k])
        highest_probs = prob_vec[-top_n:]  # pick n highest probabilities
        print('Two highest probabilities are: ', highest_probs)
        top_list = []
        for prob in highest_probs:
            top_list.append(list(trace_preds[k]).index(prob))
    print('Top-%s labels for trace are: %s' % (top_n, top_list))

    return top_list


def get_dist(all_preds):
    '''
    This function returns the distance between the pairs.
    :param l: list of list containing all the combinations fo the predictions
    : return: list containing pairs and the distances between the pairs in sorted order
    '''
    print('-'*60)
    pairs = list(itertools.product(*all_preds))
    print('total pairs generated are: ', len(pairs))
#     print('pairs are: ', pairs)

    pre_dist = {} # for saving pre-computed results of the distances 
    
    dist = []
    for i in range(len(pairs)):
        temp = pairs[i]
        print('processing pair %d - %s'%(i, temp))

        t1 = temp[0]
        t2 = temp[1]
        t3 = temp[2]
        
        k1 = str(t1) + str(t2)
        k2 = str(t2) + str(t3)
        
        if k1 in pre_dist:
            d1 = pre_dist[k1]
        else:
            d1 = get_similarity(t1, t2)
            pre_dist[k1] = d1
        
        if k2 in pre_dist:
            d2 = pre_dist[k2]
        else:
            d2 = get_similarity(t2, t3)
            pre_dist[k2] = d2

        total_dist = round(d1 + d2, 4)
#         print('total distance between the three traces: ', total_dist)
        dist.append(total_dist)
        
    print('pairs: ', pairs)
    print('distance:', dist)
    
    res = [idx for idx, val in enumerate(dist) if val in dist[:idx]]
    
    dup_indices = [] # list containing all the duplicate indices
    for di in range(len(res)):
        dup_indices.append([res[di]-1, res[di]])
    
    print('duplicate distance indices: ', dup_indices)
    
    # randomly pick indices to remove and add it to the set
    rem_indices = []
    for ri in range(len(dup_indices)):
        rem_indices.append(random.choice(dup_indices[ri]))
    
    rem_indices = set(rem_indices)
    # removing repeated elements from the list
    pairs = [v for i, v in enumerate(pairs) if i not in rem_indices]
    dist = [v for i, v in enumerate(dist) if i not in rem_indices]
    
    zipped_lists = zip(dist, pairs)
    sorted_zipped_lists = sorted(zipped_lists)
    pairs_sorted = [element for _, element in sorted_zipped_lists]
    print('pairs sorted: ', pairs_sorted)
    dist.sort()
    print('distance sorted: ', dist)
    print('-'*60)
    return pairs_sorted, dist


def get_accuracy(pairs_c, pairs_c_dist, t1_actual_label, t2_actual_label, t3_actual_label):
    '''
    This function computes the accuracy of the attack.
    :param pairs_list: list of pairs
    :param dist_list: list of distance among the pairs
    :return returns the accuracy of the attack
    '''
    temp_pairs_s = pairs_c
    dist = pairs_c_dist
    acc_1 = []
    acc_2 = []

    sel_pairs_tr_1 = []
    sel_pairs_tr_2 = []
    sel_pairs_tr_3 = []
    m = 0
    for p in range(len(dist)):
        count_1 = 0
        count_2 = 0
        count_3 = 0

        temp_pairs = temp_pairs_s[p]
        if (temp_pairs[0] not in sel_pairs_tr_1) and (temp_pairs[1] not in sel_pairs_tr_2) and (temp_pairs[2] not in sel_pairs_tr_3):
            sel_pairs_tr_1.append(temp_pairs[0])
            sel_pairs_tr_2.append(temp_pairs[1])
            sel_pairs_tr_3.append(temp_pairs[2])

            print('Updated confidence is as follows :')
            print('Trace 1: ', sel_pairs_tr_1)
            print('Trace 2: ', sel_pairs_tr_2)
            print('Trace 3: ', sel_pairs_tr_3)

            if (t1_actual_label in sel_pairs_tr_1):
                count_1 += 1
            if (t2_actual_label in sel_pairs_tr_2):
                count_2 += 1
            if (t3_actual_label in sel_pairs_tr_3):
                count_3 += 1

            accuracy = round((count_1 + count_2 + count_3)/3.0, 4)
            print('Top-%d accuracy is %f'%(m+1, accuracy))
            m += 1

            if (m == 1):
                acc_1.append(accuracy)
            if (m == 2):
                acc_2.append(accuracy)
                
    return acc_1, acc_2


def evaluation():
    top_n = 2
    acc_1 = []
    acc_2 = []

    for i in range(0, len(x_test), 3):
#     for i in range(0, 6, 3):
        print('processing pair: ', i)
#     for i in range(0, 6, 3):
        t1 = x_test[i].reshape(-1, params['features'], 1)
        t1_actual_label = list(y_test[i]).index(1)
        print('actual label of trace t1: ', t1_actual_label)
        # making predictions for trace t1
        t1_pred = model.predict(t1)
        top_list_1 = get_top_n(t1_pred, top_n)

        t2 = x_test[i+1].reshape(-1, params['features'], 1)
        t2_actual_label = list(y_test[i+1]).index(1)
        print('actual label of trace t2: ', t2_actual_label)
        # making predictions for trace t1
        t2_pred = model.predict(t2)
        top_list_2 = get_top_n(t2_pred, top_n)

        t3 = x_test[i+2].reshape(-1, params['features'], 1)
        t3_actual_label = list(y_test[i+2]).index(1)
        print('actual label of trace t3: ', t3_actual_label)
        # making predictions for trace t1
        t3_pred = model.predict(t3)
        top_list_3 = get_top_n(t3_pred, top_n)

        # generate n^3 unique combinations, where n is number of predictions
        print('+'*60)
        all_preds = [top_list_1, top_list_2, top_list_3]
        pairs_c, pairs_c_dist = get_dist(all_preds)
        print('+'*60)

        # getting the accuracy
        temp_acc_1, temp_acc_2 = get_accuracy(pairs_c, pairs_c_dist, t1_actual_label, t2_actual_label, t3_actual_label)
        acc_1.append(temp_acc_1)
        acc_2.append(temp_acc_2)
        
    return acc_1, acc_2


acc_1, acc_2 = evaluation()

acc_1_t = [item for sublist in acc_1 for item in sublist]
acc_2_t = [item for sublist in acc_2 for item in sublist]


print('Average Top-1 accuracy: ', np.mean(acc_1_t))
print('Average Top-2 accuracy: ', np.mean(acc_2_t))