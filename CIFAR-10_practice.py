# import numpy as np
# import matplotlib.pyplot as plt
# from keras.constraints import maxnorm
# import tensorflow as tf
# from tensorflow.keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # load the data – already split

# normalize input data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# one-hot encode the target values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# create the model
model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
               input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(units=256, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])

# sgb1 = SGD(learning_rate=0.001, momentum=0.8)
# sgb2 = SGD(learning_rate=0.01, momentum=0.8)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32)

scores = model.evaluate(x_test, y_test, verbose=0)
accuracy = scores[1]*100

model.summary()
print('> %.3f' % accuracy)
