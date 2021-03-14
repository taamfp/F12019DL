import numpy as np
import cv2
from PIL import ImageGrab
import win32gui
import time

width = 1023
height = 750

# Screen Recording
def screen_recorder():

    # Time Update
    last_time = time.time()
    # Recording
    while(True):
        screen_recording = ImageGrab.grab(bbox=(0,0,width,height))
        screenArray = np.array(screen_recording)
        frameConversion = cv2.cvtColor(screenArray, cv2.COLOR_BGR2RGB)

        cv2.imshow('screen recorder', np.array(frameConversion))


        print('Processing Time:', time.time() - last_time)
        last_time = time.time()

        # Key to stop recording
        if cv2.waitKey(1) == ord('t'):
            cv2.destroyAllWindows()
            break

screen_recorder()