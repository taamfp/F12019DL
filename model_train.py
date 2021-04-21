### Custom NN ###

import numpy as np
import time
from random import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import *
from tensorflow.keras import Input, regularizers, layers, models, callbacks, initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard



# Check GPU availability
deviceAvailability = tf.test.is_gpu_available()

if deviceAvailability == True:
    gpu = tf.config.list_physical_devices('GPU')
    print('GPU device: \n', gpu)

tf.config.experimental.set_memory_growth(gpu[0], True)

# Tensorboard 
model_name = 'version1-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))


# Read Data
def read_file(file):
    return np.load(file, allow_pickle=True)

# Data Sampling
def data_distribution(data_files):

    A  = []
    AZ = []
    AL = []
    AK = []
    Z  = []
    ZL = []
    ZK = []
    K  = []
    L  = []
    C  = []

    for data in data_files:
        if data[1] == [1, 0, 0, 0, 0, 0, 0, 0, 0]:
            A.append([data[0], data[1]])
        elif data[1] == [0, 1, 0, 0, 0, 0, 0, 0, 0]:
            AZ.append([data[0], data[1]])
        elif data[1] == [0, 0, 1, 0, 0, 0, 0, 0, 0]:
            AL.append([data[0], data[1]])
        elif data[1] == [0, 0, 0, 1, 0, 0, 0, 0, 0]:
            AK.append([data[0], data[1]])
        elif data[1] == [0, 0, 0, 0, 1, 0, 0, 0, 0]:
            Z.append([data[0], data[1]])
        elif data[1] == [0, 0, 0, 0, 0, 1, 0, 0, 0]:
            ZL.append([data[0], data[1]])
        elif data[1] == [0, 0, 0, 0, 0, 0, 1, 0, 0]:
            ZK.append([data[0], data[1]])
        elif data[1] == [0, 0, 0, 0, 0, 0, 0, 1, 0]:
            K.append([data[0], data[1]])
        elif data[1] == [0, 0, 0, 0, 0, 0, 0, 0, 1]:
            L.append([data[0], data[1]])
        else:
            C.append([data[0], data[1]])

    list_ = [A, AZ, AL, AK, Z, ZL, ZK, K, L, C]

    for k in list_:
        if len(A) > len(k):
            delta = len(A) - len(k)
            k = np.repeat(k, delta)
            k = k[:750]
        elif (k==A) or (len(k) > len(A)):
            k = k [:200]

    complete_data = sum(list_, [])

    shuffle(complete_data)
    
    return np.array(complete_data)

# Standardize Data
def standard_data(image):
    image /= np.std(np.mean(image, axis=0), axis=0)
    return image

# Input Size
width = 350
height = 350

# HyperParameters
learning_ = 0.000001
epochs = 100
batch_size = 75
steps_per_epoch = 18


# Model

model = Sequential()

# Input Layer
model.add(Input(shape=(height, width, 3)))

# Conv Layers
model.add(layers.Conv2D(4, (5, 5), activation='relu', kernel_initializer=initializers.GlorotNormal()))
model.add(layers.Conv2D(4, (5, 5), activation='relu'))
model.add(layers.Conv2D(4, (5, 5), activation='relu'))
model.add(layers.Conv2D(4, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(8, (5, 5), activation='relu'))
model.add(layers.Conv2D(8, (5, 5), activation='relu'))
model.add(layers.Conv2D(8, (5, 5), activation='relu'))
model.add(layers.Conv2D(8, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(124, (3, 3), activation='relu'))
model.add(layers.Conv2D(124, (3, 3), activation='relu'))
model.add(layers.Conv2D(124, (3, 3), activation='relu'))
model.add(layers.Conv2D(124, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

# Full Connected Layers
model.add(layers.Dense(125, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(75, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(70, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), bias_regularizer=tf.keras.regularizers.l2(1e-4)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(24, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), bias_regularizer=tf.keras.regularizers.l2(1e-4)))
model.add(layers.Dropout(0.5))


model.add(layers.Dense(9, activation='softmax'))

print('Custom Model Summary \n')
model.summary()


model.compile(optimizer=Adam(learning_rate = learning_, clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])

# Data Generator
def train_data_generator(batch):

    data_file = 1

    data = read_file('training_file-'+str(data_file)+'.npy')
    data = data_distribution(data)
    train_sample = 0.85*(data.shape[0])
    train_data = data[:int(train_sample)]

    shuffle(train_data)

    X = []
    Y = []

    while(True):

        for i in range(len(train_data)):

            X.append(train_data[i][0])
            Y.append(train_data[i][1])

            if len(X) > batch and i < len(train_data):

                X_train = np.array(X, dtype=np.float32).reshape(-1, height, width, 3)
                X_train = standard_data(X_train)
                Y_train = np.array(Y, dtype=np.float32)

                yield X_train, Y_train

                X = []
                Y = []

            elif i == int(train_sample)-1:
                data_file += 1
                i = 0

                if data_file > 5:
                    break



class CallbackTraining(callbacks.Callback):


    def on_train_begin(self, logs=None):
        print('Starting training ...')
        time.sleep(1)
        print('Learning Rate: ', learning_, 'Epochs: ', epochs, 'Batch_Size: ', batch_size)


    def on_epoch_end(self, epoch, logs=None):
        epoch += 1

        acc = logs['accuracy']
        val_acc = logs['val_accuracy']
        loss = logs['loss']
        val_loss = logs['val_loss']

        if epoch >= epochs:
            self.model.save('F1_model.h5')
            print('Train complete')
            self.model.stop_training = True


X_t = []
Y_t = []

test_data_file = 1
test_data = read_file('training_file-'+str(test_data_file)+'.npy')
test_data = data_distribution(test_data)
test_sample = 0.85*(test_data.shape[0])

test_ = test_data[int(test_sample):len(test_data)]

shuffle(test_)


for j in range(len(test_)):

    X_t.append(test_[j][0])
    Y_t.append(test_[j][1])

    if j == len(test_)-1:

        X_test = np.array(X_t, dtype=np.float32)
        X_test = standard_data(X_test)
        Y_test = np.array(Y_t, dtype=np.float32)

        test_data_file += 1
        test_data = read_file('training_file-'+str(test_data_file)+'.npy')

        j = 0

        if test_data_file > 5:
            break 


model.fit(train_data_generator(batch_size), epochs=epochs, batch_size=batch_size, steps_per_epoch = steps_per_epoch, 
    verbose=1,
    validation_data=(X_test.reshape(-1, height, width, 3), Y_test), 
    callbacks=[CallbackTraining(), tensorboard])