import matplotlib.pyplot as plt
from keras.utils import plot_model
from contextlib import redirect_stdout
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def __save_model_training_history_plot(history, fig_names):
    print('Saving model plots...')
    plt.figure()
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(fig_names[0])
    plt.figure()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(fig_names[1])

def __save_model_summary(model, path):
    print('Saving model summary...')
    with open(path+'summary.txt', 'w') as summary:
        with redirect_stdout(summary):
           model.summary()

def __save_history_data(history, path):
    print('Saving training/test accuracy and loss...')
    hist_dict = dict(zip(history.history.keys(), history.history.values()))
    df = pd.DataFrame.from_dict(hist_dict)
    df.to_csv(path+'acc_loss.csv')

def save_stats(model, history, path):
    __save_model_training_history_plot(history, [path+'plot_accuracy.png', path+'plot_loss.png'])
    __save_model_summary(model, path)
    __save_history_data(history, path)
    plot_model(model, to_file=path+'structure.png')

    print('Saving model configuration...')
    with open(path+'model_config.json', 'w') as json:
        json.write(model.to_json())

def save_data_info(train_datagen, test_datagen, train_set, test_set,
len_x_train, len_x_test, len_labels, path):
    print('Saving data info...')
    with open(path+'data_info.txt', 'w') as data_info:
        data_info.write('---------- Training Data ----------\n' +
        'Training sample size: {0}\n'.format(len_x_train) +
        'Number of classes/labels: {0}\n'.format(len_labels) +
        'Batch size: {0}\n'.format(train_set.batch_size) +
        '\n' +
        'Augmentation\n' +
        'Rescale: {0}\n'.format(train_datagen.rescale) +
        'Shear range: {0}\n'.format(train_datagen.shear_range) +
        'Zoom range: {0}\n'.format(train_datagen.zoom_range) +
        'Horizontal flip: {0}\n'.format(train_datagen.horizontal_flip) +
        '\n' +
        '---------- Test Data ----------\n' +
        'Test sample size: {0}\n'.format(len_x_test) +
        'Number of classes/labels: {0}\n'.format(len_labels) +
        'Batch size: {0}\n'.format(test_set.batch_size) +
        '\n' +
        'Augmentation\n' +
        'Rescale: {0}\n'.format(test_datagen.rescale) +
        'Shear range: {0}\n'.format(test_datagen.shear_range) +
        'Zoom range: {0}\n'.format(test_datagen.zoom_range) +
        'Horizontal flip: {0}\n'.format(test_datagen.horizontal_flip)
        )

def save_predictions(predictions, path):
    with open(path+'predictions.txt', 'w') as pred:
        with redirect_stdout(pred):
            for filename, letter, p in predictions:
                print('File ' + filename + ': ' + letter + ' with ' + str(p*100) + '% accuracy')

def plot_confusion_matrix(y_pred, y_test, classes, path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path+'confusion_matrix.png')

def save_histogram(path, predictions):
    plt.figure()
    plt.hist(predictions)
    plt.savefig(path+'histogram.png')