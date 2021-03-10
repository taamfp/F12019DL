import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam


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
epoch = 2500
batch_size = len(train_data)/epoch


# Model
model = Sequential()

model.add(Conv2D(shape, activation='relu', input_shape=(width, height, 3)))
model.add(Conv2D(shape, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(shape, activation='relu'))
model.add(Conv2D(shape, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(shape, activation='relu'))
model.add(Conv2D(shape, activation='relu'))
model.add(MaxPooling2D())


model.add(Flatten())

model.add(Dense(output.size, activation='softmax'))



model.summary()


model.compile(optimizer=Adam(learning_rate = learning_), loss='categorical_crossentropy', metrics=['accuracy'])