import numpy as np 
import cv2

file_ = input('Choose input file: ')
data = np.load(file_, allow_pickle=True)

while(True):

	cv2.imshow('test input', data[150][0])

	if cv2.waitKey(1) == ord('t'):
		cv2.destroyAllWindows()
		break
