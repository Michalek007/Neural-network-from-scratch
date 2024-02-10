# Neural network from scratch

Project contains implemented neural network in raw python (only with usage of Numpy) and api with example usage of this package (MNIST dataset).

## neural_network

Neural network package with implemented network, layers, activations & loss functions.
Structure:
* **core** - NeuralNetwork & Layer classes, main logic of network
* **activation** - Activation, ActivationSoftmax, ActivationReLU, ActivationTanh classes
* **loss** - Loss, LossME, LossCrossEntropy classes

## api

Flask application (rest-api) with example usage of neural_network package.
Contains trained model for MNIST dataset (detecting hand written digits) and web functionality for predicting digits from images.
Best model performed with 95.21% (tested on test dataset with 10_000 samples).

More details about api can be seen in api/README.md
