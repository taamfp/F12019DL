import numpy as np
import cv2 as opencv
import time
from PIL import ImageGrab
import os
from test_keys import pressed_released_key


# Window ImageGrab size
width = 1023
height = 750


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

def image_processing(image):

    low_level = np.array([18, 94, 140])
    up_level = np.array([48, 255, 255])

    frameBlur = opencv.GaussianBlur(image, (5, 5), 0)
    hsv = opencv.cvtColor(frameBlur, opencv.COLOR_BGR2HSV)
    mask = opencv.inRange(hsv, low_level, up_level)
    edges = opencv.Canny(mask, 75, 100)

    return edges


def keys_pressed(keys):

    move_key = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if ''.join(keys) in Keys:
        return Keys[''.join(keys)]
    else:
        return move_key



def main(file, file_index):

    print('Starting acquisition')

    # Waiting time
    time.sleep(5)

    print('Ok, go!')

    if os.path.isfile(file):
        file_index += 1
        file_data = os.path.join(path,'training_file-{}.npy'.format(file_index))

    data = []

    while(True):

        screen_recording = ImageGrab.grab(bbox=(0, 0, width, height))
        screenArray = np.array(screen_recording)
        frameConversion = opencv.cvtColor(screenArray, opencv.COLOR_BGR2RGB)
        frameConversion = opencv.resize(frameConversion, (350, 350))

        edges = image_processing(frameConversion)

        lines = opencv.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                opencv.line(frameConversion, (x1,y1), (x2,y2), (0,255,0),5)

        keys = keys_pressed(pressed_released_key())

        print(keys)

        data.append([frameConversion,keys])

        if len(data) > 10000:
            np.save(file, data)
            file_data = os.path.join(path,'training_file-{}.npy'.format(file_index))
            print('Train complete')
            data = []
            file_index += 1



if __name__ == '__main__':
    main(file_name, file_index)