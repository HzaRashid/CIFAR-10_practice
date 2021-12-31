# CIFAR-10_practice
Build a CNN to classify images containing 10 different types of objects in the popular [CIFAR-10 dataset]((https://www.cs.toronto.edu/~kriz/cifar.html).


- First time using a deep-learning library -- keras with tensorflow backend. The CIFAR-10 dataset is a great place to start since it is one of the more general datasets out there. 
- When trying out 'he_normal' (truncated normal), normal, and [he_uniform](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform), distributions to initialize the layers, the model performed best using 'he_normal'.

- After 50 epochs with the Adam optimizer, the model acheived an accuracy of 85.020%.
- After 50 epochs with the SGD optimizer (learning rate: 0.001, momentum: 0.8), the model acheived an accuracy of 79.530%.
- This outcome is consistent with the smaller learning rate that was applied to the SGD optimizer, making it better suited for more epochs. And even though the model performed better with Adam, it overfitted the data within 12-15 epochs. Wheras with SGD, the model's accuracy improved incrementally throughout the 50 epochs, showing signs of overfitting during only the last 10-15 epochs. The most appearant strategies to avoid overfitting so soon would to adjust momentum, batch size, and dropout.
