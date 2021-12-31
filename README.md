# CIFAR-10_practice
This is an Image Classification project. The goal is to build a Convolutional Neural Network (an arti) to identify 10 types of images based on the object contained in with the popular [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).


- First time using a deep-learning library -- keras with tensorflow backend. The CIFAR-10 dataset is a great place to start since it is one of the more general datasets out there. 
- When trying out 'he_normal' (truncated normal), normal, and [he_uniform](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform), distributions to initialize the layers, the model performed best using 'he_normal'.

- After 50 epochs with the Adam optimizer, the model acheived an accuracy of 85.030%. However, it overfitted the data within 12-15 epochs.
- After 50 epochs with the SGD optimizer (learning rate: 0.001, momentum: 0.8), the model acheived an accuracy of 79.530%.
- This outcome is consistent with the smaller learning rate that was applied to the SGD optimizer, making it better suited for more epochs. 
- Even though the model performed better with Adam, the model's accuracy improved incrementally throughout the 50 epochs using the SGD optimizer, showing signs of overfitting during only the last 10-15 epochs. The most apparent strategies to avoid overfitting so soon would be to adjust momentum, batch size, and dropout, to begin with.
