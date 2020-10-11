import pynput
from pynput.keyboard import Key, Listener
import time

keys_press = []
keys_released = []

def on_press(key):
    keys_press.append(key)
    print(keys_press)

def on_release(key):
    keys_released.append(key)

with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

