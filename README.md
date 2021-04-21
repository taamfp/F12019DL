# F1 2019 with Deep Learning
F1 2019 controlling cars with Deep Learning algorithms <img src="https://img.shields.io/badge/build-open%20to%suggestions-green">

<p align="center">
  <img src="Demo/demo.gif" alt="animated" />
</p>
 
Acess to keys data was obtained with [pydirectinput](https://github.com/learncodebygaming/pydirectinput) library. Check the [repository](https://github.com/learncodebygaming/pydirectinput) and [LearnCodeByGaming](https://github.com/learncodebygaming) awesome content (also on [Youtube](https://www.youtube.com/channel/UCD8vb6Bi7_K_78nItq5YITA) and official [website](https://learncodebygaming.com/)).

## Current Status

- Acess to game capture with [OpenCV](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html);
- Road Lane Line Detection with [Hough Line Transform](https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html);
- Collection of video frames data with encoded pressed and released keys(currently with .jpg, and .npy files with frames and label keys);
- Basic train and test implementation.

To collect data from the game (keys and frames), use the  ```model_input.py``` script. Individual game frames with track delimiter can be acquired with the ```data_collection.py``` script.
