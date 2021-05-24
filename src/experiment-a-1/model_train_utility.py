'''
This method contains all the utility methods required for training a model.
'''

import pandas as pd

import os

import tensorflow as tf
tf.keras.backend.clear_session()
import os
import pandas as pd
import numpy as np

# model training imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, ELU
from tensorflow.keras.initializers import glorot_uniform

import warnings
warnings.filterwarnings('ignore')


# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     print("Name:", gpu.name, "  Type:", gpu.device_type)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print(e)

def create_directory(path):
    '''
    This method creates the directory at specified path to save the CSVs for model evaluation.
    :param path: path of directory to save the CSVs
    :return:
    '''
    # creating directory for saving result for this script
    if not os.path.isdir(path):
        print('creating directory ...')
        os.makedirs(path)
        print('The results will be saved to %s'%(path))
    else:
        print('directory already available.')

def load_params():
    '''
    This method loads the parameters required for training the model
    :return: parameters for training the model
    '''
    params = {
        "optimizer": "Adamax", "epochs": 110, "learning_rate": 0.0033718063908218135, "act_fn_1": "relu",
        "act_fn_2": "relu", "act_fn_3": "softsign", "act_fn_4": "tanh", "filter_nums_1": 16, "filter_nums_2": 128,
        "filter_nums_3": 128, "filter_nums_4": 32, "kernel_size": 11, "conv_stride_size": 1, "pool_stride_size": 5,
        "pool_size": 5, "drop_rate": 0.2, "drop_rate_fc1": 0.1, "drop_rate_fc2": 0.4, "fc1_neuron": 256,
        "fc_1_act": "selu", "fc2_neuron": 1024, "fc_2_act": "selu", "batch_size": 32, "features": 2000
    }
    return params

# Defining function for loading data
def load_data_no_def(data_path, features):
    '''
    This method loads the non-defended dataset for training the model.
    :param data_path: path to directory containing data
    :param features: number of features of the dataset to be used for training
    :return: training, validation, and testing dataset
    '''
    print('reading the dataset from %s ...'%(data_path))
    # Point to the directory storing data
    dataset_dir = data_path

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    x_train = pd.read_csv(dataset_dir + 'x_train_no_def.csv')
    x_train = x_train.iloc[:, 0:features]
    y_train = pd.read_csv(dataset_dir + 'y_train_no_def.csv', header=None)
    print('Training data loaded successfully.')

    # Load validation data
    x_valid = pd.read_csv(dataset_dir + 'x_valid_no_def.csv')
    x_valid = x_valid.iloc[:, 0:features]
    y_valid = pd.read_csv(dataset_dir + 'y_valid_no_def.csv', header=None)
    print('Validation data loaded successfully.')

    # Load testing data
    x_test = pd.read_csv(dataset_dir + 'x_test_no_def.csv')
    x_test = x_test.iloc[:, 0:features]
    y_test = pd.read_csv(dataset_dir + 'y_test_no_def.csv', header=None)
    print('Testing data loaded successfully.')

    nb_classes = len(np.unique(y_test))
    print('number of classes in dataset: ', nb_classes)

    print('Data dimensions:')
    print('X training data shape: ', x_train.shape)
    print('y training data shape: ', y_train.shape)
    print('X validation data shape: ', x_valid.shape)
    print('y validation data shape: ', y_valid.shape)
    print('X testing data shape: ', x_test.shape)
    print('y testing data shape: ', y_test.shape)
    print('dataset loaded successfully!')
    print('number of classes in the dataset: ', nb_classes)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, nb_classes

# Defining function for loading data
def load_data_wt_def(data_path, features):
    '''
    This method loads the wt-defended dataset for training the model.
    :param data_path: path to directory containing data
    :param features: number of features of the dataset to be used for training
    :return: training, validation, and testing dataset
    '''
    print('reading the dataset from %s ...'%(data_path))
    # Point to the directory storing data
    dataset_dir = data_path

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    x_train = pd.read_csv(dataset_dir + 'x_train_wt_def.csv')
    x_train = x_train.iloc[:, 0:features]
    y_train = pd.read_csv(dataset_dir + 'y_train_wt_def.csv', header=None)
    print('Training data loaded successfully.')

    # Load validation data
    x_valid = pd.read_csv(dataset_dir + 'x_valid_wt_def.csv')
    x_valid = x_valid.iloc[:, 0:features]
    y_valid = pd.read_csv(dataset_dir + 'y_valid_wt_def.csv', header=None)
    print('Validation data loaded successfully.')

    # Load testing data
    x_test = pd.read_csv(dataset_dir + 'x_test_wt_def.csv')
    x_test = x_test.iloc[:, 0:features]
    y_test = pd.read_csv(dataset_dir + 'y_test_wt_def.csv', header=None)
    print('Testing data loaded successfully.')

    nb_classes = len(np.unique(y_test))
    print('number of classes in dataset: ', nb_classes)

    print('Data dimensions:')
    print('X training data shape: ', x_train.shape)
    print('y training data shape: ', y_train.shape)
    print('X validation data shape: ', x_valid.shape)
    print('y validation data shape: ', y_valid.shape)
    print('X testing data shape: ', x_test.shape)
    print('y testing data shape: ', y_test.shape)
    print('dataset loaded successfully!')
    print('number of classes in the dataset: ', nb_classes)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, nb_classes

# Defining function for loading data
def load_data_front_def(data_path, features):
    '''
    This method loads the wt-defended dataset for training the model.
    :param data_path: path to directory containing data
    :param features: number of features of the dataset to be used for training
    :return: training, validation, and testing dataset
    '''
    print('reading the dataset from %s ...'%(data_path))
    # Point to the directory storing data
    dataset_dir = data_path

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    x_train = pd.read_csv(dataset_dir + 'x_train_front_def.csv')
    x_train = x_train.iloc[:, 0:features]
    y_train = pd.read_csv(dataset_dir + 'y_train_front_def.csv', header=None)
    print('Training data loaded successfully.')

    # Load validation data
    x_valid = pd.read_csv(dataset_dir + 'x_valid_front_def.csv')
    x_valid = x_valid.iloc[:, 0:features]
    y_valid = pd.read_csv(dataset_dir + 'y_valid_front_def.csv', header=None)
    print('Validation data loaded successfully.')

    # Load testing data
    x_test = pd.read_csv(dataset_dir + 'x_test_front_def.csv')
    x_test = x_test.iloc[:, 0:features]
    y_test = pd.read_csv(dataset_dir + 'y_test_front_def.csv', header=None)
    print('Testing data loaded successfully.')

    nb_classes = len(np.unique(y_test))
    print('number of classes in dataset: ', nb_classes)

    print('Data dimensions:')
    print('X training data shape: ', x_train.shape)
    print('y training data shape: ', y_train.shape)
    print('X validation data shape: ', x_valid.shape)
    print('y validation data shape: ', y_valid.shape)
    print('X testing data shape: ', x_test.shape)
    print('y testing data shape: ', y_test.shape)
    print('dataset loaded successfully!')
    print('number of classes in the dataset: ', nb_classes)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, nb_classes

def build_model(params, nb_classes):
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