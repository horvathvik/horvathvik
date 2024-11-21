# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:15:40 2024

Train different models

@author: NaMiLAB
"""

#import only used modules!
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import iosum.dbhandler as dbhandler
import os

from pythonProject.global_vars import TRAIN_DB_PATH, TRAIN_DB_NAME, FIGURE_MODEL_HIST_PATH, MODEL_NAME, MODEL_PATH, \
    TEST_DB_PATH, TEST_DB_NAME


# Easy CID model
def cnn_model_easyCID(x_train):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(input_shape=(x_train.shape[1], 1), filters=32, kernel_size=(3), strides=(2),
                               padding='SAME', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=(2)),
        tf.keras.layers.Conv1D(input_shape=(x_train.shape[1] / 2, 32), filters=64, kernel_size=(3), strides=(2),
                               padding='SAME', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=(2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])


def train_easyCID(x_train, y_train, x_test, y_test, batch_size, epochs, optimizer='Adam'):
    model = cnn_model_easyCID(x_train)
    model.compile(optimizer=optimizer,
                  loss=['sparse_categorical_crossentropy'],
                  metrics=['accuracy'])
    model.summary()
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0.0001, patience=20, verbose=0, mode='auto',
                                                 baseline=None, restore_best_weights=True)]
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_split=0.1, verbose=0, callbacks=callback)
    if len(x_test) > 0 and len(y_test) > 0:
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return model, history

# Shuffle - deprecated
"""
def shuffle(index, x_train, y_train):
    x_new = np.empty_like(x_train)
    y_new = np.empty_like(y_train)
    index_new = np.empty_like(index)
    permutation = np.random.permutation(np.arange(0, len(x_train), 1))
    for i_old, i_new in enumerate(permutation):
        index_new[i_new] = index[i_old]
        x_new[i_new] = x_train[i_old]
        y_new[i_new] = y_train[i_old]
    return index_new, x_new, y_new
"""

def shuffle(data):
    #assumes each element of data is of the same length
    data_new = []
    for element in data:
        data_new.append(np.empty_like(element))
    permutation = np.random.permutation(np.arange(0,len(data_new[0]),1))
    for i in range(len(data)):
        for i_old, i_new in enumerate(permutation):
            data_new[i][i_new] = data[i][i_old]
    return data_new

def confusion_matrix(predicted_labels, true_labels):
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    for i, prediction in enumerate(predicted_labels):
        if prediction >= 0.5 and true_labels[i] == 1:
            true_positives += 1
        elif prediction >= 0.5 and true_labels[i] == 0:
            false_positives += 1
        elif prediction < 0.5 and true_labels[i] == 1:
            false_negatives += 1
        elif prediction < 0.5 and true_labels[i] == 0:
            true_negatives += 1
    confusion_matrix = np.array([[true_positives, false_positives], [false_negatives, true_negatives]])
    try:
        accuracy = ((true_positives + true_negatives) /
                    (true_positives + false_positives + true_negatives + false_negatives))
    except ZeroDivisionError as err:
        accuracy = np.Inf
    try:
        sensitivity = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError as err:
        sensitivity = np.Inf
    try:
        specifity = true_negatives / (true_negatives + false_positives)
    except ZeroDivisionError:
        specifity = np.Inf
    try:
        precision = true_positives / (false_positives + true_positives)
    except ZeroDivisionError as err:
        precision = np.Inf

    return confusion_matrix, accuracy, sensitivity, specifity, precision

def read_dataset_multiclass(path, db_name, clasess, train_split):
    train_data = []
    train_labels = []
    train_index = []
    test_data = []
    test_labels = []
    test_index = []

    groups = []
    for i in range(len(clasess)):
        groups.append(dbhandler.select_values(path, db_name, label=i))
        #split to train-test groups
        index = int(len(groups[i]) * train_split)
        for j in range(index):
            train_index.append(groups[i][j][0])
            train_data.append(dbhandler.convert_array(groups[i][j][2]))
            train_labels.append(groups[i][j][-1])
        for j in np.arange(index, len(groups[i]), 1):
            test_index.append(groups[i][j][0])
            test_data.append(dbhandler.convert_array(groups[i][j][2]))
            test_labels.append(groups[i][j][-1])

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    test_index, test_data, test_labels = shuffle([test_index, test_data, test_labels])
    train_index, train_data, train_labels = shuffle([train_index, train_data, train_labels])

    return train_index, train_data, train_labels, test_index, test_data, test_labels



def read_dataset(path, db_name, train_split, shuffle_data=True):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    data_positives = dbhandler.select_values(path, db_name, label=1)
    data_negatives = dbhandler.select_values(path, db_name, label=0)

    #split the data
    index = int(len(data_positives)*train_split)

    for i in range(index):
        train_data.append(dbhandler.convert_array(data_positives[i][1]))
        train_data.append(dbhandler.convert_array(data_negatives[i][1]))
        train_labels.append(data_positives[i][2])
        train_labels.append(data_negatives[i][2])

    for i in np.arange(index,len(data_positives),1):
        test_data.append(dbhandler.convert_array(data_positives[i][1]))
        test_data.append(dbhandler.convert_array(data_negatives[i][1]))
        test_labels.append(data_positives[i][2])
        test_labels.append(data_negatives[i][2])

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    test_data, test_labels = shuffle(test_data, test_labels)
    train_data, train_labels = shuffle(train_data, train_labels)

    return train_data, train_labels, test_data, test_labels


def save_figure(file_path, name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path+'/'+name+".png")
    return 0

index_train, x_train, y_train, index_test, x_test, y_test = read_dataset_multiclass(TRAIN_DB_PATH,TRAIN_DB_NAME,['under 24 ppm','in-between','over 240 ppm'],0.8)


# Create and train model
epochs = 500
batch_size = 100
w_len = 50
dw = int(w_len / 2)


model, history = train_easyCID(x_train, y_train, x_test, y_test, batch_size, epochs, optimizer='Adam')

plt.figure()
plt.plot(np.arange(1, len(history.epoch) + 1, 1), history.history["loss"], 'b.', label='Test loss')
plt.plot(np.arange(1, len(history.epoch) + 1, 1), history.history["val_loss"], 'b--', label='Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
save_figure(FIGURE_MODEL_HIST_PATH,"training_loss_"+MODEL_NAME)

plt.figure()
plt.plot(np.arange(1, len(history.epoch) + 1, 1), history.history["accuracy"], 'b.', label='Test accuracy')
plt.plot(np.arange(1, len(history.epoch) + 1, 1), history.history["val_accuracy"], 'b--', label='Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
save_figure(FIGURE_MODEL_HIST_PATH,"training_accuracy_"+MODEL_NAME)

# Evaluate Model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Loss: {}".format(loss))
print("Accuracy :{}".format(acc))

prediction = model.predict(x_test)

correct_prediction = 0
incorrect_prediction = 0

for i, pred in enumerate(prediction):
    y_pred = np.argmax(pred)
    y_true = y_test[i]
    if y_true == y_pred:
        correct_prediction += 1
    else:
        incorrect_prediction += 1

print("Correct predictions: {}/{}".format(correct_prediction, correct_prediction+incorrect_prediction))
plt.show()


#saving the model and the test files
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
model.save(MODEL_PATH+"/"+MODEL_NAME+".keras")
data_toadd = []
for i in range(len(x_test)):
    data_toadd.append((int(index_test[i]), x_test[i], int(y_test[i])))

dbhandler.create_db(TEST_DB_PATH,TEST_DB_NAME,['spectra'],[['ID','data', 'label']])
dbhandler.add_values_batch(TEST_DB_PATH,TEST_DB_NAME,"spectra",data_toadd)




