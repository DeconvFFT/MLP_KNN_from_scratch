# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [Saumya Hetalbhai Mehta] -- [mehtasau]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding
from math import sqrt

class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def xavier_initialization(self):
        """
        Function to initialise weights when we have a tanh or sigmoid activation function.
        Used Xavier Initialization
        """
        n_features = self._X.shape[1]
        n_outputs = self._y.shape[1]

        val = 1.0/sqrt(n_features)
        
        # initialise hidden weights
        self._h_weights = np.random.uniform(-val, val, (n_features, self.n_hidden))
       
        self._h_bias = np.ones((1, self.n_hidden))

        val = 1.0/sqrt(n_features)

        # initialise output weights
        self._o_weights = np.random.uniform(-val, val, (self.n_hidden, n_outputs))
       
        self._o_bias = np.ones((1, n_outputs))
        
    def relu_initialization(self):
        """
        Function to initialise weights when we have a relu activation function.
        Used He Weight Initialization
        """
        n_features = self._X.shape[1]
        n_outputs = self._y.shape[1]
        std = sqrt(2.0 / n_features)

        # Initialise hidden weights
        self._h_weights = np.random.randn((n_features, self.n_hidden))*std
        self._h_bias = np.ones((1, self.n_hidden))

        # Initialise output weights
        self._o_weights = np.random.randn((self.n_hidden, n_outputs))*std
        self._o_bias = np.ones((1, n_outputs))

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._X = X
        self._y = one_hot_encoding(y)
        
        np.random.seed(42)
        self.relu_initialization() if self.hidden_activation =="relu" else self.xavier_initialization()

        #raise NotImplementedError('This function must be implemented by the student.')


    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._initialize(X, y)


        for epoch in range(self.n_iterations):

            # Forward pass computation 
            # activation(Wx+b) for hidden
            hidden_input = self._X.dot(self._h_weights)+ self._h_bias
            hidden_output = self.hidden_activation(hidden_input)

            # activation(Wx+b) for output layer
            output_layer_input = hidden_output.dot(self._o_weights) + self._o_bias
            y_pred = self._output_activation(output_layer_input)
            celoss = np.sum(self._loss_function(self._y, y_pred))
            self._loss_history.append(celoss)
            # if epoch%20 == 0:
            #     print(f"CELOSS: {celoss}")
            # Backward pass, update weights and biases
            
            # gradients for output layer's input
            del_out_layer_inp = (y_pred-self._y) * self._output_activation(output_layer_input, derivative=True)
            del_out_w = hidden_output.T.dot(del_out_layer_inp)
            del_out_b = np.sum(del_out_layer_inp, axis = 0, keepdims=True)

            self._o_weights -= self.learning_rate * del_out_w
            self._o_bias -= self.learning_rate * del_out_b
            
            # gradients for hidden layer input
            del_hidden_layer_inp = del_out_layer_inp.dot(self._o_weights.T) * self.hidden_activation(hidden_input, derivative=True)
            del_hidden_w = self._X.T.dot(del_hidden_layer_inp)
            del_hidden_b = np.sum(del_hidden_layer_inp, axis = 0, keepdims=True)
            

            
            self._h_weights -= self.learning_rate * del_hidden_w
            self._h_bias -= self.learning_rate * del_hidden_b

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        hidden_inp = X.dot(self._h_weights) + self._h_bias
        hidden_output = self.hidden_activation(hidden_inp)
        output_layer_input = hidden_output.dot(self._o_weights) + self._o_bias
        y_pred = self._output_activation(output_layer_input)
        y_pred = np.argmax(y_pred, axis =1)
        return y_pred
        #raise NotImplementedError('This function must be implemented by the student.')
