import numpy as np
import time
import tensorflow as tf
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

width = 1050
height = 900

# Parameters
learning_ = 0.001
epochs = 100
train_sample = 0.8*(data.shape[0])

train_data = data[:int(-train_sample)]
test_data = data[int(-train_sample):]


batch_size = 50

# Model
model = Sequential()

model.add(Input(shape=(height, width, 3)))

model.add(layers.Conv2D(4, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(len(data[0][1]), activation='softmax'))


model.summary()


model.compile(optimizer=Adam(learning_rate = learning_), loss='categorical_crossentropy', metrics=['accuracy'])


class CallbackTraining(callbacks.Callback):

    def on_train_begin(self, logs=None):
        print('Starting training ', learning_, epochs, batch_size)


    def on_epoch_end(self, epoch, logs=None):

        self.model.save('F1_model.h5')

        acc = logs['accuracy']
        val_acc = logs['val_accuracy']
        loss = logs['loss']
        val_loss = logs['val_loss']

        if epoch >= epochs:
            print('Train complete')
            self.model.stop_training = True

X = []
Y = []

X_t = []
Y_t = []


for i in range(len(train_data)):
    X.append(train_data[i][0])
    X_train = np.array(X, dtype=np.float32)
    Y.append(train_data[i][1])
    Y_train = np.array(Y, dtype=np.float32)

for j in range(len(train_data)):
    X_t.append(train_data[j][0])
    X_test = np.array(X_t, dtype=np.float32)
    Y_t.append(train_data[j][1])
    Y_test = np.array(Y_t, dtype=np.float32)


model.fit(X_train.reshape(-1, height, width, 3), Y_train, epochs=epochs, batch_size=batch_size, verbose=1, 
    validation_data=(X_test.reshape(-1, height, width, 3), Y_test), 
    callbacks=[CallbackTraining()])