import tensorflow
from tensorflow.keras.utils import *
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

deviceAvailability = tf.test.is_gpu_available()

if deviceAvailability == True:
	gpu = tf.test.gpu_device_name()
	print('GPU device: \n', gpu)


input_shape = image.array

model = Sequential()

model.add(Conv2D(shape, activation='relu', input_shape=input_shape))
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

learning_rate = value

model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])