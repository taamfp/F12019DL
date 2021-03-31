import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import *
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet


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
def read_file(file):
    return np.load(file, allow_pickle=True)


width = 350
height = 350
channels = 3

# Parameters
learning_ = 0.00001
epochs = 500
batch_size = 170
steps_per_epoch = 18


mobile_layers = MobileNet(include_top=False, weights='imagenet', input_shape=(width, height, channels))
 

model = Sequential(mobile_layers)

for layer in model.layers:
	layer.trainable = False

model.add(layers.Flatten())
model.add(layers.Dense(9, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(learning_rate = learning_, epsilon=0.0000001, clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy'])


def train_data_generator(batch):

    data_file = 1

    data = read_file('training_file-'+str(data_file)+'.npy')

    train_sample = 0.90*(data.shape[0])
    train_data = data[:int(train_sample)]
    train_data = np.random.shuffle(train_data)

    X = []
    Y = []

    while(True):

        for i in range(len(train_data)):

            X.append(train_data[i][0])
            Y.append(train_data[i][1])

            if len(X) > batch and i < len(train_data):
                X_train = np.array(X, dtype=np.float32).reshape(-1, height, width, 3)
                Y_train = np.array(Y, dtype=np.float32)

                yield X_train, Y_train

                X = []
                Y = []

            elif i == int(train_sample)-1:
                data_file += 1
                i = 0

                if data_file > 7:
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
            self.model.save('F1_pretrained_model.h5')
            print('Train complete')
            self.model.stop_training = True



X_t = []
Y_t = []

test_data_file = 1
test_data = read_file('training_file-'+str(test_data_file)+'.npy')
test_sample = 0.90*(test_data.shape[0])

test_ = test_data[int(test_sample):len(test_data)]
test_ = np.random.shuffle(test_)

np.random.shuffle(test_)


for j in range(len(test_)):

    X_t.append(test_[j][0])
    Y_t.append(test_[j][1])

    if j == len(test_)-1:

        X_test = np.array(X_t, dtype=np.float32)
        Y_test = np.array(Y_t, dtype=np.float32)

        test_data_file += 1
        test_data = read_file('training_file-'+str(test_data_file)+'.npy')

        j = 0

        if test_data_file > 7:
            break 


model.fit(train_data_generator(batch_size), epochs=epochs, batch_size=batch_size, steps_per_epoch = steps_per_epoch, 
    verbose=1,  
    validation_data=(X_test.reshape(-1, height, width, 3), Y_test), 
    callbacks=[CallbackTraining()])