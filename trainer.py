'''
From: https://keras.io/getting-started/sequential-model-guide/#multilayer-perceptron-mlp-for-multi-class-softmax-classification
'''
#Imports
#Initialize neural network model
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
from keras.optimizers import SGD, Adadelta
from keras.preprocessing import image
from img_loader import load_images
import tensorflow as tf
from keras.utils import multi_gpu_model, to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os

#Custom python scripts
from prediction_chooser import get_highest_predictions
#For preprocessing images
from keras.preprocessing.image import ImageDataGenerator
from exp_stats import save_stats, save_data_info, save_histogram
from exp_stats import plot_confusion_matrix
from img_loader import load_images

# Importing augmentation functions
from augmentationsettings import augmentationSettings

#Building CNN model

def __get_image_generator(path, X_train, X_test, y_train, y_test):
    #Preprocessing images to avoid overfitting
    '''
    Performing augmentation of images.
    Preparing test data.
    Data directory name will be the label.
    '''
    # Augmentation settings from document
    train_datagen = augmentationSettings("basicAug")
    test_datagen = ImageDataGenerator(rescale=1./255)

    #Create image dirs if not exists
    train_dir = '{0}training_images'.format(path)
    test_dir = '{0}test_images'.format(path)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)


    train_set = train_datagen.flow(X_train, y_train,
                                   save_to_dir=train_dir,
                                   batch_size=5)
    test_set = test_datagen.flow(X_test, y_test,
                                 save_to_dir=test_dir,
                                 batch_size=5)
 
    return (train_datagen, test_datagen, train_set, test_set)

def __create_model():
    #Creates sequential network
    model = Sequential()

    #Convolution
    '''
    Adding convolutional layer
    param 1: number of filters
    param 2: filter shape (matrix)
    param 3: input shape 200x200 image, and type, 3 = RGB
    param 4: activation function, relu = rectifier function
    '''
    model.add(Conv2D(32, (5,5), strides=1, input_shape=(200, 200, 3), activation='relu'))
    model.add(Conv2D(32, (3,3), strides=1, input_shape=(200, 200, 3), activation='relu'))
    model.add(Conv2D(32, (3,3), strides=1, input_shape=(200, 200, 3), activation='relu'))


    #Pooling
    '''
    Reducing image size as much as possible
    Helps reducing number of nodes in the layers
    Trying to reduce complexity without loosing performance
    '''
    model.add(AveragePooling2D(pool_size=(2,2), strides=2))

    #Convolution
    model.add(Conv2D(64, (3,3), strides=1, input_shape=(100, 100, 3), activation='relu'))
    model.add(Conv2D(64, (3,3), strides=1, input_shape=(100, 100, 3), activation='relu'))

    #Pooling
    model.add(AveragePooling2D(pool_size=(2,2), strides=2))

    #Convolution
    model.add(Conv2D(128, (2,2), strides=1, input_shape=(50, 50, 3), activation='relu'))

    #Pooling
    model.add(AveragePooling2D(pool_size=(2,2), strides=2))

    #Convolution
    model.add(Conv2D(256, (3,3), strides=1, input_shape=(25, 25, 3), activation='relu'))

    #Pooling
    model.add(AveragePooling2D(pool_size=(2,2), strides=2))

    #Convolution
    model.add(Conv2D(512, (3,3), strides=1, input_shape=(13, 13, 3), activation='relu'))

    #Pooling
    model.add(AveragePooling2D(pool_size=(2,2), strides=2))

    #Convolution
    model.add(Conv2D(512, (3,3), strides=1, input_shape=(7, 7, 3), activation='relu'))

    #Pooling
    model.add(AveragePooling2D(pool_size=(2,2), strides=2))

    #Fully Connected Layer
    '''
    Make images into continous vector.
    Converting 2-D image pixel matrices to
    one dimensional vector.
    Keras can do the flattening automatically
    '''
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(200, activation='relu'))

    #Output layer
    model.add(Dense(5, activation='softmax')) #research why softmax is used

    #Compiling CNN model

    '''
    optimizer: choosing stochastic gradient decent algorithm
    loss: choosing the loss function
    metrics: choosing the performance metric
    '''
    #sgd: Stochastic Gradient Decent
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def __train_model(save_path, train_set, test_set):
    model = __create_model()

    #Fitting data to model

    epochs = 2
    '''
    steps_per_epoch: number of training images
    epochs: Number of times the network will be trained
    on every training sample.
    validation_steps: number of test images
    '''
    history = model.fit_generator(train_set,
                        steps_per_epoch=2,
                        epochs=epochs,
                        validation_data=test_set,
                        validation_steps=2)


    print('Saving model...')
    model.save(save_path+'model.hdf5')
    save_stats(model, history, save_path)

    return model


def get_model(save_path, create):
    #Preprocess images in samples
    seed = 7
    X, y, label_map = load_images('../images/')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    train_datagen, test_datagen, train_set, test_set = __get_image_generator(save_path,
    X_train, X_test, y_train_cat, y_test_cat)

    if save_path.endswith('.hdf5'):
            save_path = save_path[:-5]
    if create:
        model = __train_model(save_path, train_set, test_set)
    else:
        model = load_model(save_path+'.hdf5')

    save_data_info(train_datagen, test_datagen, train_set, test_set,
    len(X_train), len(X_test), len(label_map), save_path)

    #Evaluate model predictions
    print('Generating confusion matrix...')
    predictions = model.predict_generator(test_set, steps=2)
    y_pred = np.argmax(predictions, axis=1)
    y_true = y_test[:10]
    print(y_pred)
    print(y_true)
    plot_confusion_matrix(y_pred, y_true, label_map.values(), save_path)
    save_histogram(save_path, predictions)


    return model
