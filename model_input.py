import numpy as np
import cv2 as opencv
import time
from PIL import ImageGrab
import os
from Test_Keys import pressed_released_key



# Window size
width = 1050
height = 900



path = 'C:/Users/Utilizador/Documents/GitHub/F1withML'

file_index = 1
file_name = os.path.join(path,'training_file-{}.npy'.format(file_index))

Keys = {
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

move_key = [0, 0, 0, 0, 0, 0, 0, 0, 0]


def keys_pressed(keys):

	move_key = [0, 0, 0, 0, 0, 0, 0, 0, 0]

	if ''.join(keys) in Keys:
		return Keys[''.join(keys)]
	else:
		return move_key



def main(file, file_index):

	print('Starting acquisition')

	time.sleep(5)

	if os.path.isfile(file):
		file_index += 1
		file_data = os.path.join(path,'training_file-{}.npy'.format(file_index))

	data = []

	while(True):

		screen_recording = ImageGrab.grab(bbox=(0, 0, width, height))
		screenArray = np.array(screen_recording)
		frameConversion = opencv.cvtColor(screenArray, opencv.COLOR_BGR2RGB)

		keys = keys_pressed(pressed_released_key())

		data.append([frameConversion,keys])

		if len(data) > 100:
		   np.save(file, data)
		   data = []
		   file_index += 1
		   file_data = os.path.join(path,'training_file-{}.npy'.format(file_index))


if __name__ == '__main__':
	main(file_name, file_index)