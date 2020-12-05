import pynput
from pynput.keyboard import Key, Listener
import time

keys_press = []
keys_released = []

def on_press(key):
	print(key)

def on_release(key):
    keys_released.append(key)
    print(keys_released)

with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()