# CIFAR-10_practice
Build a CNN to classify 10 different types of objects in the popular CIFAR-10 dataset.
- The [dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- Helpful resources –> Project specific:   ["How to Develop a CNN from Sctrach for CIFAR-10 Photo Classification"](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/), ["Deep Learning with CIFAR-10 Image Classification"](https://towardsdatascience.com/deep-learning-with-cifar-10-image-classification-64ab92110d79) -- General: [tensorflow.keras library guide](https://www.tensorflow.org/api_docs/python/tf/keras), [Guide for CNNS](https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480)

- This was my first time working keras/tensorflow. After learning the approriate methods in the keras/tensorflow libraries, and the base model implementations provided in the links, I tried builing one of my own, which is the one in this repo. 
- The base model in this repo performed best using he_normal (truncated normal) distribition compared to normal and [he_uniform](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform) distributions to initialize the kernels.
- After 50 epochs with the Adam optimizer, the model acheived an accuracy of 85.020%.
- After 50 epochs with the SGD optimizer (learning rate: 0.001, momentum: 0.8), the model acheived an accuracy of 79.530%.
- This outcome seems normal considering the smaller learning rate with SGD, making it better suited for more epochs. Although the model performed better with Adam, its accuracy plateaued within 25-30 epochs, so it overfitted the data quickly. Wheras with SGD, the model's accuracy improved incrementally throughout the 50 epochs.
