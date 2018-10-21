import os
import cv2
import pandas as pd
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Hyperparameters
batch_size = 128
epochs = 3

def get_model():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu', input_shape=(128, 128, 1) ))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.20))

    model.add(Dense(1, activation = 'sigmoid'))

    return model


def train():

    # DATA

    # GRAB POSITIVE AND NEGATIVE IMAGES

    dataset_path = 'data/hand-no-hand/'
    pos_data_dir = os.path.join(dataset_path, 'positive')
    neg_data_dir = os.path.join(dataset_path, 'negative')    

    pos_img = [f for f in os.listdir(pos_data_dir) if '.jpg' in f]
    neg_img = [f for f in os.listdir(neg_data_dir) if '.jpg' in f]
    
    num_pos = len(pos_img)
    threshold_pos = int(num_pos * 0.8) # 20% as test set
    train_pos, test_pos = pos_img[:threshold_pos], pos_img[threshold_pos:]
    
    num_neg = len(neg_img)
    threshold_neg =  int(num_neg * 0.8)
    train_neg, test_neg = neg_img[:threshold_neg], neg_img[threshold_neg:]

    X_train = []
    y_train = []
    for filename in train_pos:
        image = cv2.imread(os.path.join(pos_data_dir, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        array = image.flatten()
        X_train.append(image)
        y_train.append(1)
    for filename in train_neg:
        image = cv2.imread(os.path.join(neg_data_dir, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        array = image.flatten()
        X_train.append(image)
        y_train.append(0)

    X_test = []
    y_test = []
    for filename in test_pos:
        image = cv2.imread(os.path.join(pos_data_dir, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        array = image.flatten()
        X_test.append(array)
        y_test.append(1)
    for filename in test_neg:
        image = cv2.imread(os.path.join(neg_data_dir, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        array = image.flatten()
        X_test.append(array)
        y_test.append(0)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = X_train.reshape(X_train.shape[0], 128, 128, 1)
    X_test = X_test.reshape(X_test.shape[0], 128, 128, 1)

    # MODEL

    model = get_model()

    print('Model Summary')
    print(model.summary())

    model.compile(loss = keras.losses.binary_crossentropy, 
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'])

    history = model.fit(X_train, 
                        y_train, 
                        validation_data = (X_test, y_test), 
                        epochs=epochs, 
                        batch_size=batch_size)

    print('Validation accuracy:', history.history['val_acc'][-1])

    # SAVE

    print('Saving model')
    model.save('binary_hand_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model


def hand_detection(image, model):

    print('Classifying hand or not!')
    image =  image.reshape(image.shape[0], image.shape[1], 1)
    images = np.array([ image ])
    return model.predict(images)


if __name__ == '__main__':
    if not os.path.isfile('binary_hand_model.h5'):
        print('Train the binary classification model')
        train()
    else:
        print('Model already trained')