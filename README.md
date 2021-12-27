# CIFAR-10_practice
Build a CNN to classify 10 different types of objects in the popular CIFAR-10 dataset.
- The [dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- Helpful resources: Project specific:   ["How to Develop a CNN from Sctrach for CIFAR-10 Photo Classification"](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/), ["Deep Learning with CIFAR-10 Image Classification"](https://towardsdatascience.com/deep-learning-with-cifar-10-image-classification-64ab92110d79) -- General: [tensorflow.keras library guide](https://www.tensorflow.org/api_docs/python/tf/keras), [keras library guide](https://keras.io/guides/), [Guide for CNNs](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

- I wanted to gain hands on experience with a library built for deep learning since I had never used one before. The CIFAR-10 dataset was a great place to start since it is one of the more general datasets out there. Most of the models built to 'solve' the dataset use the keras/tensorflow library, so thats the one I used in my first project using a deep learning library.
- After stuyding the prerequisites for this project (from the above links), I tried builing a model of my own, which is the one in this repo. 
- The model in this repo uses 'he_normal' (truncated normal) distribution to initialize the kernel at every convolutional 2D layer and every dense layer (except for the output) because the base model performed best using this distribution compared to normal and [he_uniform](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform). 
- After 50 epochs with the Adam optimizer, the model acheived an accuracy of 85.020%.
- After 50 epochs with the SGD optimizer (learning rate: 0.001, momentum: 0.8), the model acheived an accuracy of 79.530%.
- This outcome seems normal considering the smaller learning rate with SGD, making it better suited for more epochs. Although the model performed better with Adam, its accuracy plateaued within 25-30 epochs, so it overfitted the data quickly. Wheras with SGD, the model's accuracy improved incrementally throughout the 50 epochs.
