# CIFAR-10_practice
This is an Image Classification practice project using the popular [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). It contains 60,000 images (32x32, rgb) divided into 10 classes (categories) based on the object contained in them.

The goal is to build a [Convolutional Neural Network](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) to identify the 10 classes of images. 


- This was my first time using a deep-learning library -- keras with tensorflow backend. The CIFAR-10 dataset is a great place to start since it is one of the more general datasets out there, and the data is already organized. So most of the time spent on this project goes to optimizing the model and learning the libraries.

- Important to note: there are a lot of standard image classification models that have been applied to this dataset - many of them can be found [here](https://paperswithcode.com/sota/image-classification-on-cifar-10)
- For the purpose of learning the keras library, I tried making a model from stratch using common techniques for image classification (e.g., stacking convolutional layers, using batch normalization, adding dropout layers), which can be found in [helpful_resources](https://github.com/HzaRashid/CIFAR-10_practice/blob/main/helpful_resources.pdf). Some unique aspects of the model in the main branch of this repo: initializing the kernels using 'he_normal' (truncated normal) distribution, not stacking the input layer, not stacking the last convolutional layer, as well as setting the strides to (2,2) in the last max pooling layer.

## Brief report:
### The model in Patch 1:
  - This model was more conventional than the one in main branch. All convolutional layers were stacked except for the last one, a max pooling layer was added after each stacked pair of convolutional layers, and also after the last convolutional layer.
  - It was trained with 50 epochs.
  - Using the Adam optimizer, the model acheived an accuracy of 85.030%.
  - Using the SGD optimizer (learning rate: 0.001, momentum: 0.8), the model acheived an accuracy of 79.530%.
  - Although the model performed better with Adam, it overfitted the data within 12-15 epochs. With SGD, the model's accuracy improved more consistently throughout training, showing signs of overfitting during only the last 10-15 epochs. The most apparent strategies to avoid overfitting so soon (with either optimizer) would be to adjust momentum, batch size, and dropout, to begin with.

### The model in main branch:
  - Trained using the Adam optimizer and 5 epochs. It achieved an accuracy of 81.350%


#### Will be adding accuracy plots to this readme soon.
