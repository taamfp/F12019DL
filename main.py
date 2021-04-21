import os
import numpy as np
import time
import cv2 as opencv
import tensorflow as tf
from PIL import ImageGrab
from tensorflow.keras import models
from model_input import image_processing
import pydirectinput

# Inference test with CPU (change to 1 for GPU inference)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_file = 'F1_model_v2.h5'
model = models.load_model(model_file)

# Input Size
width = 1023
height = 750


def main():

    while(True):

        t_time = time.time()

        car_screen = ImageGrab.grab(bbox=(0, 0, width, height))
        frameConversion = opencv.cvtColor(np.array(car_screen), opencv.COLOR_BGR2RGB)

        edges = image_processing(frameConversion)

        lines = opencv.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap = 50)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                opencv.line(frameConversion, (x1,y1), (x2,y2), (0,255,0),5)

        frameConversion = opencv.resize(frameConversion, (350, 350))

        y_pred = model.predict(frameConversion.reshape(-1, 350, 350, 3)).flatten()

        for i in range(len(y_pred)):
            if i == np.argmax(np.array(y_pred)):
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        y_pred = [int(i) for i in y_pred]

        if np.argmax(y_pred) == 0:
            pydirectinput.keyDown(str('A').lower())
            pydirectinput.keyUp(str('Z').lower())
            pydirectinput.keyUp(str('K').lower())
            pydirectinput.keyUp(str('L').lower())

        elif np.argmax(y_pred) == 1:
            pydirectinput.keyDown(str('A').lower())
            pydirectinput.keyDown(str('Z').lower())
            pydirectinput.keyUp(str('K').lower())
            pydirectinput.keyUp(str('L').lower())

        elif np.argmax(y_pred) == 2:
            pydirectinput.keyDown(str('A').lower())
            pydirectinput.keyDown(str('L').lower())
            pydirectinput.keyUp(str('Z').lower())
            pydirectinput.keyUp(str('K').lower())

        elif np.argmax(y_pred) == 3:
            pydirectinput.keyDown(str('A').lower())
            pydirectinput.keyDown(str('K').lower())
            pydirectinput.keyUp(str('Z').lower())
            pydirectinput.keyUp(str('L').lower())

        elif np.argmax(y_pred) == 4:
            pydirectinput.keyDown(str('Z').lower())
            pydirectinput.keyUp(str('A').lower())
            pydirectinput.keyUp(str('K').lower())
            pydirectinput.keyUp(str('L').lower())

        elif np.argmax(y_pred) == 5:
            pydirectinput.keyDown(str('Z').lower())
            pydirectinput.keyDown(str('L').lower())
            pydirectinput.keyUp(str('A').lower())
            pydirectinput.keyUp(str('K').lower())

        elif np.argmax(y_pred) == 6:
            pydirectinput.keyDown(str('Z').lower())
            pydirectinput.keyDown(str('K').lower())
            pydirectinput.keyUp(str('A').lower())
            pydirectinput.keyUp(str('L').lower())

        elif np.argmax(y_pred) == 7:
            pydirectinput.keyDown(str('K').lower())
            pydirectinput.keyUp(str('A').lower())
            pydirectinput.keyUp(str('Z').lower())
            pydirectinput.keyUp(str('L').lower())

        elif np.argmax(y_pred) == 8:
            pydirectinput.keyDown(str('L').lower())
            pydirectinput.keyUp(str('A').lower())
            pydirectinput.keyUp(str('Z').lower())
            pydirectinput.keyUp(str('K').lower())

        else:
            pydirectinput.keyUp(str('A').lower())
            pydirectinput.keyUp(str('Z').lower())
            pydirectinput.keyUp(str('K').lower())
            pydirectinput.keyUp(str('L').lower())

        print('Processing Time:', time.time() - t_time)
        print('Movement: ', y_pred)


if __name__ == '__main__':
    main()