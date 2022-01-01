from keras.datasets import cifar10
import cifar10_models

(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # load the data – already split

# normalize inputs
x_train = x_train / 255.0
x_test = x_test / 255.0

# create the model
model = cifar10_models.model_1()

# opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)