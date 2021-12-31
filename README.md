# CIFAR-10_practice
This is an Image Classification project using the popular [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). It contains 60,000 images (32x32, rgb) divided into 10 classes (categories) based on the object contained in them.

The goal of this project is to build a [Convolutional Neural Network](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) to identify the 10 classes of images. 


- First time using a deep-learning library -- keras with tensorflow backend. The CIFAR-10 dataset is a great place to start since it is one of the more general datasets out there. 

- After 50 epochs with the Adam optimizer, the model acheived an accuracy of 85.030%. However, it overfitted the data within 12-15 epochs.
- After 50 epochs with the SGD optimizer (learning rate: 0.001, momentum: 0.8), the model acheived an accuracy of 79.530%.
- This outcome is consistent with the smaller learning rate that was applied to the SGD optimizer, making it better suited for more epochs. 
- Although the model performed better with Adam, its accuracy improved more consistently throughout the 50 epochs using the SGD optimizer, showing signs of overfitting during only the last 10-15 epochs. The most apparent strategies to avoid overfitting so soon would be to adjust momentum, batch size, and dropout, to begin with.
