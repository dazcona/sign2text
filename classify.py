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
num_classes = 25
epochs = 20

def get_model():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu', input_shape=(28, 28, 1) ))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.40))

    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.20))

    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.20))

    model.add(Dense(num_classes, activation = 'softmax'))

    return model


def train():

    # DATA

    dataset_path = 'data/language/'
    train_data = pd.read_csv(os.path.join(dataset_path, 'sign_mnist_train.csv'))
    test_data = pd.read_csv(os.path.join(dataset_path, 'sign_mnist_test.csv'))

    X_train = train_data.iloc[:, 1:].values
    X_test = test_data.iloc[:, 1:].values

    y_train = train_data.iloc[:, :1].values.flatten()
    y_test = test_data.iloc[:, :1].values.flatten()

    X_train = X_train / 255.
    X_test = X_test / 255.

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    # MODEL

    model = get_model()

    print('Model Summary')
    print(model.summary())

    model.compile(loss = keras.losses.categorical_crossentropy, 
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'])

    history = model.fit(X_train, 
                        y_train_encoded, 
                        validation_data = (X_test, y_test_encoded), 
                        epochs=epochs, 
                        batch_size=batch_size)

    print('Validation accuracy:', history.history['val_acc'][-1])

    # SAVE

    print('Saving model')
    model.save('recognition_sign_model.h5')  # creates a HDF5 file 'recognition_sign_model.h5'
    del model  # deletes the existing model


def classify_gesture(image, model):

    print('Classifying hand gesture!')
    images = np.array([ image ])
    return model.predict(images)


if __name__ == '__main__':
    
    if not os.path.isfile('recognition_sign_model.h5'):
        print('Train the classification model')
        train()
    else:
        print('Model already trained')