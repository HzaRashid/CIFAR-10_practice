# CIFAR-10_practice
Build a CNN to classify 10 different types of objects in the popular CIFAR-10 dataset.
- First time working with keras/tensorflow
- The [dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- Helpful resources: Project specific: [1](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/), [2](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/), [3](https://medium.com/@jayramchaudhury20/project-on-image-classification-on-cifar-10-dataset-94db0ff6baf5), 
General: [4](https://www.tensorflow.org/api_docs/python/tf/keras), [5](https://en.wikipedia.org/wiki/Truncated_normal_distribution)

- misc: At first, I followed the above links closely as this was my first deep dive into a deep learning project. I tried out the base models provided in those links, learning a lot concepts such as kernel constraints, maxnorm, and the types of distributions used to initialize the kernels (weights). I also spent a considerable amount of time learning the keras and tensorflow libraries! 

- Once I felt comfortable enough with the library and base models provided in the links, I tried builing one of my own, which is the one in this repo.

- There are 10 layers in total. The first 8 layers can be thought of as pairs (i.e. 4 pairs), where the 1st layer in a pair is 2D convolutional, and the 2nd is max pooling of shape (2,2). In the first pair, the 2D convolutional layer has 32 filters, and in each of the following pairs, the 2D convolutional layer has double the amount of filters of the one in the previous pair (i.e. 32, 64, 128). As well, The convolutional layer in each of the 4 pairs has kernel of shape (3, 3), initialized using 'truncated normal distribition' (written as 'he_normal' in the argument of model.fit), which is one of the many concepts I learned during this practice project! 
- Moreover, the 9th layer is dense, with 256 units, and instead of having the kernel initialized to 'he_normal', the argument is left blank (and set to 'glorot_uniform' by default), and a kernel constraint is set to a maxnorm of 3. Finally, the output layer is dense, with 10 units (i.e. the 10 different types of objects), using softmax activation.
