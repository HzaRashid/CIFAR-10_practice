from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization


def model_main():

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(units=256, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add((Dense(10, activation='softmax')))

    return model


def model_patch_1():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(units=256, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add((Dense(10, activation='softmax')))

    return model


def model_patch_2():

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(units=256, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add((Dense(10, activation='softmax')))

    return model