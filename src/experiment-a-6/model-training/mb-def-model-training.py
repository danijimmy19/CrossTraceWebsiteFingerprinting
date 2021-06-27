#!/usr/bin/env python
# coding: utf-8
'''
to run the script execute the following command:
python mb-def-model-training.py /path/to/save/model/model-name.h5 /path/to/dataset
The dataset should be in CSV format. There are 6 CSV files. The datain each of these files is as follows:
x_train.csv -- contains the traffic traces for training the model
y_train.csv -- contains the labels corresponding to traffic traces used for training the model
x_valid.csv -- contains the traffic traces for validation
y_valid.csv -- contains the labels corresponding to traffic traces used for validating the model
x_test.csv  -- contains the traffic traces for testing the model
y_test.csv  -- contains the labels corresponding to traffic traces used for testing the model
'''

import sys
import matplotlib.pyplot as plt
from model_train_utility import *
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping

if len(sys.argv) != 3:
    print('please provide appropriate number of arguments')
    sys.exit(-1)

# path to save the model
MODEL_PATH = sys.argv[1]
# path to the dataset used for training and testing
DATA_PATH = sys.argv[2]

print('model will be saved at: %s'%(MODEL_PATH))

# load parameters of the model
p = load_params()
print('parameters of the model are: %s', p)

# load the dataset required for training the model
# path to non-defended dataset to be used for training and testing model
x_train, y_train, x_valid, y_valid, x_test, y_test, nb_classes = load_data_mb_def(DATA_PATH, p["features"])

# Changing dataframes to numpy
x_train = x_train.to_numpy()
x_train = x_train[:, :,np.newaxis]
x_valid = x_valid.to_numpy()
x_valid = x_valid[:, :,np.newaxis]
x_test = x_test.to_numpy()
x_test = x_test[:, :,np.newaxis]

verbose = 1 # Display logs while training
nb_epochs = p["epochs"] # Number of epochs for training
optimizer = optimizers.Adamax(lr=p['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-09, decay = 0.0)
length = x_train.shape[1] # Packet sequence length
print('length of input vector: %d'%(length))
input_shape = (length, 1)

# Converting output variables to categorical
y_train = to_categorical(y_train, nb_classes)
y_valid = to_categorical(y_valid, nb_classes)
y_test = to_categorical(y_test, nb_classes)

print('building and compiling a model for training non-defended boyang dataset ...')
model = build_model(p, nb_classes)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
print('model built and compiled successfully!')

call_backs = []
call_backs.append(EarlyStopping(monitor='val_accuracy', mode='max', patience=5))

print('Training model ...')
history = model.fit(x_train, y_train,
                    batch_size=p['batch_size'],
                    epochs=nb_epochs,
                    verbose=verbose,
                    validation_data=(x_valid, y_valid),
                   callbacks=call_backs)


# summarize history for accuracy
fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# summarize history for loss
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

model.save(MODEL_PATH)
history_df = pd.DataFrame.from_dict(history.history)
print('model saved successfully!')

# Start evaluating model with training data
score_train = model.evaluate(x_train, y_train, verbose=0)
print("Training accuracy: %f"%(score_train[1]))
# Start evaluating model with validation data
score_valid = model.evaluate(x_valid, y_valid, verbose=0)
print("validation accuracy: %f"%(score_valid[1]))

# Start evaluating model with testing data
score_test = model.evaluate(x_test, y_test, verbose=0)
print("Testing accuracy: %f"%(score_test[1]))

# Top N prediction
top_N = 2 # Specify top_N = n; n is top-n prediction
print("Start evaluating Top-%s Accuracy"%top_N)
result = model.predict(x_test, verbose=2)
count = 0
total = 0
actual_y = y_test
for i in range(len(result)):
    prob_vec = sorted(result[i])
    highest_probs = prob_vec[-top_N:] # pick two highest probabilities in softmax
    top_list = []
    for prob in highest_probs:
        top_list.append(list(result[i]).index(prob))
    actual_label = list(actual_y[i]).index(1) # convert from one-hot-vector back to actual label
    if actual_label in top_list:
        count = count + 1
    total = total + 1
    
print("Top-%s Accuracy: %f "%(top_N, float(count)/total))
print('program execution completed!')