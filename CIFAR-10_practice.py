from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # load the data – already split

# normalize input data
x_train = x_train / 255.0
x_test = x_test / 255.0

# create the model
model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
               input_shape=(32, 32, 3)),
        BatchNormalization(),

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
        Dropout(0.2),

        Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(units=256, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)
