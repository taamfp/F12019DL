import tensorflow
from tensorflow.keras.utils import *
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

deviceAvailability = tf.test.is_gpu_available()

if deviceAvailability == True:
	gpu = tf.test.gpu_device_name()
	print('GPU device: \n', gpu)

model_name = 'version1-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

width = 1050
height = 900

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

learning_rate = value

model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])