#Imports
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os

#Custom python scripts
from prediction_chooser import get_highest_predictions
from img_loader import load_images
from trainer import get_model
from exp_stats import save_predictions
from live_prediction import predict_live_video

def choose_model(load):
    root = Tk()
    root.lift()
    if load.lower() == 'y':
        path = askopenfilename()
        root.destroy()
        return (path, get_model(path,create=False))
    else:
        print('Choose model save path (empty for no saving)')
        path = askdirectory()
        root.destroy()

        if path is None or path.replace(' ','') == '':
            print('Empty path...')
            return

        if not os.path.exists(path):
            os.makedirs(path)

        return (path, get_model(path+'/', create=True))

load = input('Load existing model? (Y/n) ')

path, model = choose_model(load)

#Testing on our own images
if path.endswith('.hdf5'):
    path = path[:-5]

test = input('Test on images from folder? (Y/n) ')

if test.lower() == 'y':
    root = Tk()
    root.lift()
    test_image_path = askdirectory()
    root.destroy()
    test_images = load_images(test_image_path)
    predictions = get_highest_predictions(model, test_images)
    save_predictions(predictions, path)


live = input('Start live video prediction? (Y/n) ')

if live.lower() == 'y':
    predict_live_video(model)
else:
    print('Session ended...')

