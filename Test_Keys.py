import win32api as wapi


charGame = ["\b", "A", "Z", "K", "L", "G"]

def pressed_released_key():
    list_keys = []
    for key in charGame:
        if wapi.GetAsyncKeyState(ord(key)):
            list_keys.append(key)
    return list_keys