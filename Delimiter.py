import numpy as np
import cv2
from PIL import ImageGrab
import win32gui
import time
import matplotlib.pyplot as plt

width = 1023
height = 750

def image_processing(image):

    low_level = np.array([18, 94, 140])
    up_level = np.array([48, 255, 255])

    frameBlur = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(frameBlur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_level, up_level)
    edges = cv2.Canny(mask, 75, 100)

    return edges

while(True):

        # Time Update
        time_ = time.time()

        screen_recording = ImageGrab.grab(bbox=(0,0,width,height))
        screenArray = np.array(screen_recording)
        frameConversion = cv2.cvtColor(screenArray, cv2.COLOR_BGR2RGB)

        edges = image_processing(frameConversion)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap = 50)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frameConversion, (x1,y1), (x2,y2), (0,255,0),5)

        cv2.imshow("screen recorder", np.array(frameConversion))
        cv2.imshow("edges", edges)

        #print('Processing Time:', time.time() - last_time)

        print(time.time() - time_)

        # Key to stop recording
        if cv2.waitKey(1) == ord('t'):
            cv2.destroyAllWindows()
            break