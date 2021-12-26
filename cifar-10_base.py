# import numpy as np
# import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # load the data – already split

# note - the classes are: 'airplane' (0),  'automobile' (1), 'bird' (2), 'cat' (3), 'deer' (4),
#                         'dog' (5), 'frog' (6),  'horse' (7),  'ship' (8), 'truck' (9)

# normalize input data –
# 1 –> set the entries to single precision floats,
# 2 –> divide by 255.0 (max range of colour)
# then each entry lies between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train)  # converts target values – from integers that range from 0 to 9,
y_test = to_categorical(y_test)    # to vectors with 10 entries, where all entries are 0 except for
                                   # the one that corresponds to the object detected in the image
                                   # (which will have 1 in its entry)


# create the model
model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), padding='same',
               activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)),
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal'),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal'),
        Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(units=128, activation='relu', kernel_initializer='normal'),
        Dense(10, activation='softmax')
    ])


lrn_rate = 0.001
sgd = SGD(learning_rate=lrn_rate, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32)

scores = model.evaluate(x_test, y_test, verbose=0)
accuracy = scores[1]*100
print('> %.3f' % accuracy)