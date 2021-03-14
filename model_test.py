import os
import numpy as np
import time
import cv2 as opencv
import tensorflow as tf
from PIL import ImageGrab
from tensorflow.keras import models
from model_input import image_processing
import pydirectinput

# Model test with CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_file = 'F1_model.h5'
model = models.load_model(model_file)

width = 350
height = 350


keys = {
	'A':  [1, 0, 0, 0, 0, 0, 0, 0, 0],
	'AZ': [0, 1, 0, 0, 0, 0, 0, 0, 0],
	'AL': [0, 0, 1, 0, 0, 0, 0, 0, 0],
	'AK': [0, 0, 0, 1, 0, 0, 0, 0, 0],
	'Z':  [0, 0, 0, 0, 1, 0, 0, 0, 0],
	'ZL': [0, 0, 0, 0, 0, 1, 0, 0, 0],
	'ZK': [0, 0, 0, 0, 0, 0, 1, 0, 0],
	'K':  [0, 0, 0, 0, 0, 0, 0, 1, 0],
	'L':  [0, 0, 0, 0, 0, 0, 0, 0, 1],
}


def bot_test():

	while(True):

		t_time = time.time()

		car_screen = ImageGrab.grab(bbox=(0, 0, width, height))
		frame = opencv.cvtColor(np.array(car_screen), opencv.COLOR_BGR2RGB)

		frameConversion = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
		frameConversion = opencv.resize(frameConversion, (width, height))

		edges = image_processing(frameConversion)

		lines = opencv.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)

		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]
				opencv.line(frameConversion, (x1,y1), (x2,y2), (0,255,0),5)


		y_pred = model.predict(frameConversion.reshape(-1, width, height, 3)).flatten()

		for i in range(len(y_pred)):
			if i == np.argmax(np.array(y_pred)):
				y_pred[i] = 1
			else:
				y_pred[i] = 0

		y_pred = [int(i) for i in y_pred]

		time.sleep(0.3)


		for i, j in keys.items():
			if j == y_pred:
				print(j)
				pydirectinput.keyDown(str(i).lower())
				pydirectinput.keyUp(str(i).lower())
				if len(str(i))>1:
					pydirectinput.keyDown(str(i).lower()[0])
					pydirectinput.keyDown(str(i).lower()[1])
					pydirectinput.keyUp(str(i).lower()[1])
					pydirectinput.keyUp(str(i).lower()[1])
				break


		print('Processing Time:', time.time() - t_time)

bot_test()