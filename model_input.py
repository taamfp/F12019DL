import numpy as np
import cv2 as opencv
import time
from Delimiter import image_processing
from PIL import ImageGrab
import os
from Test_Keys import pressed_released_key


# Window size

width = 1050
height = 900

# Change path

path = 'C:/Users/Utilizador/Documents/GitHub/F1withML/'


# Keys array

a  =  [1, 0, 0, 0, 0, 0, 0, 0, 0]
az =  [0, 1, 0, 0, 0, 0, 0, 0, 0]
al =  [0, 0, 1, 0, 0, 0, 0, 0, 0]
ag =  [0, 0, 0, 1, 0, 0, 0, 0, 0]
z  =  [0, 0, 0, 0, 1, 0, 0, 0, 0]
zl =  [0, 0, 0, 0, 0, 1, 0, 0, 0]
zg =  [0, 0, 0, 0, 0, 0, 1, 0, 0]
l  =  [0, 0, 0, 0, 0, 0, 0, 1, 0]
g  =  [0, 0, 0, 0, 0, 0, 0, 0, 1]


file_index = 1

file_data = os.path.join(path,'training_data-1.npy')


if os.path.isfile(file_data):
	file_index += 1
	file_data = os.path.join(path,'training_data-{}.npy'.format(file_index))


def keys_pressed(key_array):

	keys_on = [0, 0, 0, 0, 0, 0, 0, 0, 0]

	if 'A' in key_array:
		keys_on = a
	elif 'A' and 'Z' in key_array:
		keys_on = az
	elif 'A' and ',' in key_array:
		keys_on = al
	elif 'A' and '.' in key_array:
		keys_on = ag
	elif 'Z' in key_array:
		keys_on = z
	elif 'Z' and ',' in key_array:
		keys_on = zl
	elif 'Z' and '.' in key_array:
		keys_on = zg
	elif ',' in key_array:
		keys_on = l
	else:
		keys_on = g

	return keys_on



def main(file, file_index):

	file = file_data
	data = []

	while(True):

		screen_recording = ImageGrab.grab(bbox=(0, 0, width, height))
		screenArray = np.array(screen_recording)
		frameConversion = opencv.cvtColor(screenArray, opencv.COLOR_BGR2RGB)

		edges = image_processing(frameConversion)
		lines = opencv.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]
				lines = opencv.line(frameConversion, (x1, y1), (x2, y2), (0, 255, 0), 5)

		keys = keys_pressed(pressed_released_key)

		data.append([frameConversion,keys])

		if len(data) > 1000:
		   np.save(file, data)
		   data = []
		   file_index += 1
		   file_data = os.path.join(path,'training_data-{}.npy'.format(file_index))
		else:
			np.save(file, data)


if __name__ == "__main__":
	main(file_data, file_index)