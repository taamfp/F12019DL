# Win32 API
import win32api as wapi


# Char game
charGame = ["\b", "A", "Z", "K", "L", "G"]


# Check pressed keys
def pressed_released_key():
    list_keys = []
    for key in charGame:
        if wapi.GetAsyncKeyState(ord(key)):
            list_keys.append(key)
    return list_keys