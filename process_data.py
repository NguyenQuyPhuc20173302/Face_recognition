from tensorflow.keras import applications, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, ZeroPadding2D, MaxPooling2D, LeakyReLU, \
    BatchNormalization
from tensorflow.keras.optimizers import SGD
import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 224


def Model_(number_label):
    base_model = applications.resnet.ResNet50(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 1))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(number_label, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=SGD(lr=0.001, decay=0.000001, momentum=0.9),
                  metrics=['accuracy'])
    return model


def read_image(file_path):
    img = cv2.imread(file_path, 0)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def take_data():
    DIR = 'image'
    list_name = []
    x_train = []
    y_train = []
    i = 0
    for name in os.listdir(DIR):
        folder = os.path.join(DIR, name)
        list_name.append(name)
        for img in os.listdir(folder):
            image = os.path.join(folder, img)
            anh = read_image(image)
            x_train.append(np.array(anh))
            y_train.append(i)
        i += 1
    x_train = np.array([i for i in x_train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
    x_train /= 255.0
    y_train = to_categorical(y_train)
    return list_name, x_train, y_train


def model_pridect():
    list_name, x_train, y_train = take_data()
    model = Model_(len(list_name))
    model.fit(x_train, y_train, epochs=10, batch_size=16)
    return model


'''list_name, x_train, y_train = take_data()
model = Model_(len(list_name))
from sklearn.model_selection import train_test_split
(X_train, X_test), (Y_train, Y_test) = train_test_split(x_train, y_train, test_size=0.3, random_state=10)
model.fit(X_train, Y_train, epochs=10, batch_size=4, validation_data= (X_test, Y_test))'''