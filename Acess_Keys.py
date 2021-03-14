# Pydirectinput sourcecode: https://github.com/learncodebygaming/pydirectinput

import os
import numpy as np
import time
import pydirectinput
import win32api
import win32gui


sta = time.time()

# Press keys
def running():
    while time.time()-sta < 18:
        pydirectinput.keyDown('a')

running()

# Record keys
list = []