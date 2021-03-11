import os
import numpy as np
import time
import cv2
import tensorflow as tf
import PIL
from tensorflow.keras import models
from model_input import *
import pydirectinput

# Model test with CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


model_file = 'F1_model.h5'
model = models.load_model(model_file)


keys = {
	'a':  [1, 0, 0, 0, 0, 0, 0, 0, 0],
	'az': [0, 1, 0, 0, 0, 0, 0, 0, 0],
	'al': [0, 0, 1, 0, 0, 0, 0, 0, 0],
	'ak': [0, 0, 0, 1, 0, 0, 0, 0, 0],
	'z':  [0, 0, 0, 0, 1, 0, 0, 0, 0],
	'zl': [0, 0, 0, 0, 0, 1, 0, 0, 0],
	'zk': [0, 0, 0, 0, 0, 0, 1, 0, 0],
	'k':  [0, 0, 0, 0, 0, 0, 0, 1, 0],
	'l':  [0, 0, 0, 0, 0, 0, 0, 0, 1],
}


def bot_test():

	while(True):

		t_time = time.time()

		car_screen = PIL.ImageGrab.grab(bbox=(0, 0, width, height))
		frame = cv2.cvtColor(np.array(car_screen), cv2.COLOR_BGR2RGB)

		y_pred = model.predict(frame.reshape(-1, height, width, 3)).flatten()
		print(y_pred)

		for i, j in Keys.items():
			if j in y_pred:
				pydirectinput.keyDown(i)
				break

		print('Processing Time:', time.time() - t_time)

bot_test()