#!/usr/bin/env python
# coding: utf-8

'''
To run this script execute the following command:
python m-2-glove-cosine.py /path/to/mb-def-model.h5 /path/to/paired/data

/path/to/mb-def-model.h5 is the path to CNN model trained on MockingBird defended dataset
/path/to/paired/data is the path to the paired dataset generated from the test data
'''

import os
import sys
import string
import random
import re

import tensorflow as tf
tf.keras.backend.clear_session()
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import optimizers
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, ELU
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical, plot_model

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) != 3:
    print('please provide appropriate number of arguments.')
    sys.exit(-1)

# path of the CNN model
# CNN_PATH = '../webfp-crossed-fingerprints/closed-world/models/wt-def-250/wt-def-keyword-100-classes-2000-250-direction-model.h5'
CNN_PATH = sys.argv[1]
# path of the dataset
# data_dir = '../webfp-crossed-fingerprints-data/closed-world/wt-def-keyword/wt-analysis-two-traces-250/paired-data/wt-def-keyword-2-paired.csv'
data_dir = sys.argv[2]

# path of the NLP model
NLP_MODEL_PATH = '../../../../data/nlp/models/glove.6B.200d.txt'
# path for text mapping
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


# reading Glove word embeddings into a dictionary with "word" as key and values as word vectors
embeddings_index = dict()

with open(NLP_MODEL_PATH) as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

def most_similar(doc_id,similarity_matrix,matrix):
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix==doc_id:
            continue
        return similarity_matrix[doc_id][ix]

def glove_similarity(documents):
    '''
    This function computes the similarity between the two documents using Glove + Cosine.
    '''
    
    documents_df=pd.DataFrame(documents,columns=['documents'])
    
    # removing special characters and stop words from the text
    stop_words_l=stopwords.words('english')
    documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )
    
    tfidfvectoriser=TfidfVectorizer()
    tfidfvectoriser.fit(documents_df.documents_cleaned)
    tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)
    
    tfidf_vectors=tfidf_vectors.toarray()
    
    # tokenize and pad every document to make them of the same size
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(documents_df.documents_cleaned)
    tokenized_documents=tokenizer.texts_to_sequences(documents_df.documents_cleaned)
    tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=200,padding='post')
    vocab_size=len(tokenizer.word_index)+1
    
    # creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
    embedding_matrix=np.zeros((vocab_size,200))

    for word,i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    # calculating average of word vectors of a document weighted by tf-idf
    document_embeddings=np.zeros((len(tokenized_paded_documents),200))
    words=tfidfvectoriser.get_feature_names()

    # instead of creating document-word embeddings, directly creating document embeddings
    for i in range(documents_df.shape[0]):
        for j in range(len(words)):
            document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors[i][j]

    pairwise_similarities=cosine_similarity(document_embeddings)
#     pairwise_differences=euclidean_distances(document_embeddings)
        
    sim = most_similar(0,pairwise_similarities,'Cosine Similarity')
    return sim


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
    
    l1: top-n predictions of trace 1
    l2: top-n predictions for trace 2
    return: list of pairs and their corresponding similarity
    """
    sim_lst = []
    pairs = []
    lst1 = l1
#     print('predictions from trace 1 are: ', lst1)
    
    lst2 = l2
#     print('predictions from trace 2 are: ', lst2)
    
    # creating pairs for trace 1
    for i in range(len(lst1)):
        web_1_temp = text_map.loc[text_map['encoded-label'] == lst1[i]]
        text_1_path = web_1_temp.iloc[0][2]
        with open(text_1_path, 'r') as f1:
            text_1 = f1.read()
            text_1 = re.sub(r'[^\w\s]', '', str(text_1).lower().strip())
            lst_text = text_1.split()

            if stop_words is not None:
                lst_text = [word for word in lst_text if word not in stop_words]

            lst_text = [sb.stem(word) for word in lst_text]
            lst_text = [lem.lemmatize(word) for word in lst_text]
#             text_1 = lst_text
            text_1 = " ".join(lst_text)
        
        for j in range(len(lst2)):
#             print('pairs are: ', lst1[i], '-', lst2[j])
            web_2_temp = text_map.loc[text_map['encoded-label'] == lst2[j]]
            text_2_path = web_2_temp.iloc[0][2]
            with open(text_2_path, 'r') as f2:
                text_2 = f2.read()
                text_2 = re.sub(r'[^\w\s]', '', str(text_2).lower().strip())
                lst_text = text_2.split()

                if stop_words is not None:
                    lst_text = [word for word in lst_text if word not in stop_words]

                lst_text = [sb.stem(word) for word in lst_text]
                lst_text = [lem.lemmatize(word) for word in lst_text]
#                 text_2 = lst_text
                text_2 = " ".join(lst_text)
            
            messages = [text_1, text_2]
            
            sim = glove_similarity(messages)
            print('similarity between %d and %d is %f'%(lst1[i], lst2[j], sim))
            pairs.append((lst1[i], lst2[j]))
            sim_lst.append(sim)
    return pairs, sim_lst

top_n = 2
sim_traces = []
acc_1 = []
acc_2 = []

# making predictions using NLP
for i in range(0, len(x_test), 2):
# for i in range(0, 8, 2):
    print('processing pair: ', i)
    t1 = x_test[i].reshape(-1, params['features'], 1)
    t1_actual_label = list(y_test[i]).index(1)
    print('actual label of trace t1: ', t1_actual_label)
    # making predictions for trace t1
    t1_pred = model.predict(t1)
    
    # getting top-n predictions for trace 1
    for k in range(len(t1_pred)):
        prob_vec = sorted(t1_pred[k])
        highest_probs = prob_vec[-top_n:] # pick two highest probabilities
        print('Two highest probabilities are: ', highest_probs)
        top_list_1 = []
        for prob in highest_probs:
            top_list_1.append(list(t1_pred[k]).index(prob))
    print('Top-%s labels for trace 1 are: %s'%(top_n, top_list_1))
    
    t2 = x_test[i+1].reshape(-1, params['features'], 1)
    t2_actual_label = list(y_test[i+1]).index(1)
    print('actual label of trace t2: ', t2_actual_label)
    # making preditions for trace t2
    t2_pred = model.predict(t2)
    
    # getting top-n predictions for trace 2
    for k in range(len(t2_pred)):
        prob_vec = sorted(t2_pred[k])
        highest_probs = prob_vec[-top_n:] # pick two highest probabilities
        print('Two highest probabilities are: ', highest_probs)
        top_list_2 = []
        for prob in highest_probs:
            top_list_2.append(list(t2_pred[k]).index(prob))
    print('Top-%s labels for trace 1 are: %s'%(top_n, top_list_2))
    
    pairs, similarity = get_similarity(top_list_1, top_list_2)
    
    if 0.0 in similarity:
        s_zero = similarity.index(0.0)
        del similarity[s_zero]
        del pairs[s_zero]
        
    print('+'*80)
    zipped_lists = zip(similarity, pairs)
    sorted_zipped_lists = sorted(zipped_lists, reverse = True)
    pairs_sorted = [element for _, element in sorted_zipped_lists]
    print(pairs_sorted)
    similarity.sort(reverse = True)
    print(similarity)
    sel_pairs_tr_1 = []
    sel_pairs_tr_2 = []
    dist = []
    m = 0
    for p in range(len(similarity)):
        count_1 = 0
        count_2 = 0

        temp_pairs = pairs_sorted[p]
        if (temp_pairs[0] not in sel_pairs_tr_1) and (temp_pairs[1] not in sel_pairs_tr_2):
            sel_pairs_tr_1.append(temp_pairs[0])
            sel_pairs_tr_2.append(temp_pairs[1])

            print('Updated confidence is as follows :')
            print('Trace 1: ', sel_pairs_tr_1)
            print('Trace 2: ', sel_pairs_tr_2)

            if (t1_actual_label in sel_pairs_tr_1):
                count_1 += 1
            if (t2_actual_label in sel_pairs_tr_2):
                count_2 += 1
            accuracy = round((count_1 + count_2)/2.0, 4)
            print('Top-%d accuracy is %f'%(m+1, accuracy))
            m += 1

            if (m == 1):
                acc_1.append(accuracy)
            if (m == 2):
                acc_2.append(accuracy)


        if ((temp_pairs[0] in sel_pairs_tr_1) and (temp_pairs[1] in sel_pairs_tr_2)):
            dist.append(similarity[p])

    print('+'*80)
    print('-'*80)

print('Average top-1 accuracy is: ', np.mean(acc_1))
print('Average top-2 accuracy is: ', np.mean(acc_2))