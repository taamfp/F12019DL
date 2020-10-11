import numpy as np
import cv2 as opencv
import time
from Delimiter import image_processing
from PIL import ImageGrab
import os

# Change path according to your case
path = 'C:/Users/Utilizador/Documents/GitHub/F1withML/data'

width = 1000
height = 700

# Recording frames with delta
i_frame = 0

start = time.time()

# Currently at delta
while(True):

    i_frame += 1
    delta_t = time.time()-start
    screen_recording = ImageGrab.grab(bbox=(0, 0, width, height))
    screenArray = np.array(screen_recording)
    frameConversion = opencv.cvtColor(screenArray, opencv.COLOR_BGR2RGB)

    edges = image_processing(frameConversion)

    lines = opencv.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lines = opencv.line(frameConversion, (x1, y1), (x2, y2), (0, 255, 0), 5)

    if delta_t > 3:
        start = time.time()
        opencv.imwrite(os.path.join(path, 'Image_' + str(i_frame) + '.jpg'), lines)

    start = time.time()

    if opencv.waitKey(1) == ord('t'):
        opencv.destroyAllWindows()
        break