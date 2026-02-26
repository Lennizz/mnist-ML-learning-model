# mnist-ML-learning-model
This is my first attempt at Machine Learning. 

## Methods

*Version 1*
I used genetic evolution. For training optimization I used mini-batching to get quicker training time and multi-threading to run multiple tests at once. 

*Version 2*
I used gradient descent. Still uesed mini-batching with multi-threading, but instead of randomly choosen 200 images, it went through all images in shuffeled 200 batches. 

Used ReLU as the non-linear function instead of sigmoid and He Initialization instead of random floats between -1.0 and 1.0

## Results

*Version 1*
I obtained a result of 90.13% using a model with no hidden layers, I have not tried to run it for any longer periods of time using a model with more hidden layers. One improvement is that I would like to test different configurations of the Neural Network to see if this improves the accuracy.

*Version 2*
I obtained a result of 97.89% using a model with 2 hidden layers, 128 and 64 nodes respectivly. 
