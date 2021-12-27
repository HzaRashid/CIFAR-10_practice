# CIFAR-10_practice
Build a CNN to classify 10 different types of objects in the popular CIFAR-10 dataset.
- The [dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- Helpful resources: Project specific:   [link 1](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/), 
                                         [link 2](https://data-flair.training/blogs/image-classification-deep-learning-project-python-keras/) |
                     General:   [link 3](https://www.tensorflow.org/api_docs/python/tf/keras),
                                [link 4](https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480)

- This was my first time working keras/tensorflow. So at first, I followed the above links closely, trying out the base models provided in those links, and learning a lot about optimizing the kernels, and the tradeoffs between SGD and Adam as the gradient optimizer.
- After learning the approriate methods in the library, and the base model implementations, I tried builing one of my own, which is the one in this repo. 
- The base model performed best using he_normal (truncated normal) distribition compared to normal and [he_uniform](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform) distributions to initialize the kernels.
- In 50 epochs with the Adam optimizer, the model acheived an accuracy of 85.020%.
