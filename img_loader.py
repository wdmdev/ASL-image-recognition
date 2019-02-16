import numpy as np
from keras.preprocessing import image
import os
from os import listdir
from os.path import isfile, join

'''
Preprocesses a single frame to make it ready for
the keras model prediction
:param frame: image matrix
:type frame: matrix
:returns
'''
def preprocess_frame(frame):
        f = image.img_to_array(frame)
        f = np.expand_dims(f, axis=0)
        return f
'''
test_image: image to be tested by the CNN.
Convert to 64x64 resolution to work with model.
predict() will give binary 1 or 0 for A or B.
'''
def load_images(load_path):
    x_input = []
    y_output = []
    label_map = {}

    for idx, category in enumerate(listdir(load_path)):
        path = join(load_path,category)
        for img in listdir(path):
            im = image.load_img(join(path,img), target_size=(200,200))
            im = image.img_to_array(im)
            im = np.expand_dims(im, axis=0)
            x_input.append(im)
            y_output.append(idx)
        label_map[idx] = category

    x_input = np.concatenate(x_input, axis=0)
    return (x_input, y_output, label_map)
