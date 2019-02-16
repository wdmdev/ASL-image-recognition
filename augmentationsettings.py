from keras.preprocessing.image import ImageDataGenerator

def augmentationSettings(augmentation):
    #No augmentation. Only rescaling.
    if augmentation == "noAug":
        train_datagen = ImageDataGenerator(rescale=1./255)
        return train_datagen

    # Traditional augmentation. Rotation, shearing, zooming. No color changes. Translation. 
    elif augmentation == "basicAug":
        train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    rotation_range=25,
                                    zoom_range=0.2,
                                    width_shift_range=0.25,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
        return train_datagen
    
    #Implementing channel shifting, white noise, brightness adjustments.
    elif augmentation == "colorAug":
        train_datagen = ImageDataGenerator(rescale=1./255,
                                    channel_shift_range=0.2,
                                    brightness_range=(0, 1))
        return train_datagen

    #Standardization, normalization 1
    elif augmentation == "stdNormAug1":
        train_datagen = ImageDataGenerator(rescale=1./255,
                                    samplewise_center=True,
                                    samplewise_std_normalization=True)
        return train_datagen
    
    #Standardization, normalization 2
    elif augmentation == "stdNormAug2":
        train_datagen = ImageDataGenerator(rescale=1./255,
                                    featurewise_center=True,
                                    featurewise_std_normalization=True)
        return train_datagen
    
    #Standardization, normalization 3
    elif augmentation == "stdNormAug3":
        train_datagen = ImageDataGenerator(rescale=1./255,
                                    samplewise_center=True,
                                    featurewise_center=True,
                                    samplewise_std_normalization=True,
                                    featurewise_std_normalization=True)
        return train_datagen

    #Combination of best results. Horisontal flip is false because of right hand
    elif augmentation == "combo1":
        train_datagen = ImageDataGenerator(rescale=1./255,
                            shear_range=0.2,
                            rotation_range=25,
                            zoom_range=0.2,
                            width_shift_range=0.15,
                            height_shift_range=0.15,
                            horizontal_flip=False,
                            brightness_range=(0.2, 0.8),
                            fill_mode='nearest')
        return train_datagen