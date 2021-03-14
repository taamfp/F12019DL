import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import *
from tensorflow.keras import layers, models
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import pydirectinput


# Tensorflow setup

# Check GPU availability
deviceAvailability = tf.test.is_gpu_available()

if deviceAvailability == True:
    gpu = tf.config.list_physical_devices('GPU')
    print('GPU device: \n', gpu)

tf.config.experimental.set_memory_growth(gpu[0], True)

# Tensorboard
model_name = 'version1-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))


# Data
file_ = input('Choose input file: ')
data = np.load(file_, allow_pickle=True)

width = 350
height = 350

# Parameters
learning_ = 0.00001
epochs = 150
batch_size = 150
steps_per_epoch = 15

train_sample = 0.85*(data.shape[0])

train_data = data[:int(train_sample)]
test_data = data[int(train_sample):len(data)]


# Model
model = Sequential()

model.add(Input(shape=(height, width, 3)))

model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))


model.add(layers.Flatten())
model.add(layers.BatchNormalization())

model.add(layers.Dense(125, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), bias_regularizer=tf.keras.regularizers.l2(1e-4)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(125, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), bias_regularizer=tf.keras.regularizers.l2(1e-4)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(125, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), bias_regularizer=tf.keras.regularizers.l2(1e-4)))
model.add(layers.Dropout(0.5))


model.add(layers.Dense(len(data[0][1]), activation='softmax'))


model.summary()


model.compile(optimizer=Adam(learning_rate = learning_, epsilon=0.0001, clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])


def train_data_generator(batch):
    X = []
    Y = []
    while(True):
        for i in range(len(train_data)):
            X.append(train_data[i][0])
            Y.append(train_data[i][1])
            if len(X) > batch:
                X_train = np.array(X, dtype=np.float32).reshape(-1, height, width, 3)
                Y_train = np.array(Y, dtype=np.float32)
                yield X_train, Y_train
                X = []
                Y = []


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


for j in range(len(test_data)):
    X_test = np.array(X_t.append(test_data[j][0]), dtype=np.float32)
    Y_test = np.array(Y_t.append(test_data[j][1]), dtype=np.float32)


model.fit(train_data_generator(batch_size), epochs=epochs, batch_size=batch_size, steps_per_epoch = steps_per_epoch, 
    verbose=1,  
    validation_data=(X_test.reshape(-1, height, width, 3), Y_test), 
    callbacks=[CallbackTraining()])