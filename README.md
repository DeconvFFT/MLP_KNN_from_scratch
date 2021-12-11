# a4
# K Nearest Neighbours

## Fit

This function sets the value of X and Y of the classifier as X and Y values that are passed using parameters.

## Predict

This function predicts the class target value for an input X. It tries to find the best possible class label based on the value on which most of the K neighbours of that input agree the most. 


## Inverse Distance vs Uniform weighing

1.) Inverse Distance Weighing: 

With Inverse Distance Weighing the labels in the neighbourhood will have share that is inversely proportional to their distance from the current observation. i.e nearest neighbours will have the highest share.  For example if the distances of 5 neighbours (0,1,1,0,0) are [3,5,6,2,4], the weight of 1 would be: 1/5 + 1/6 = 11/30 and weight of 0 would be 1/3 + 1/2+ 1/4 = 13/12. So, we would choose the label as 0 in this case.

2.) Uniform Weighing: 

With Uniform Weighing, each class gets as equal weight. And so, the only thing we consider is the class on which most of the neighbours agree.

## Distance Functions 

1.) Eucledian Distance

2.) Manhatten distance


# Multi-Layer Perceptron

## Forward Pass

For the forward pass, we would take the sum of the product of weights and inputs for one single neuron in the hidden layerand then finally add a bias to it. So, sum(wixi) + b or dot(w,x)+ b. We apply an activation function to this input to hidden layer so the output of hidden layer would be z = activation(hidden input). Now, we pass again take the dot product of z with weights of output layer and add the output bias to it. So, dot(z, wout) + biasout. And finally, we apply an output activation function (softmax in this example) to get inference from our MLP.  We would also encouter some loss/ errors during this step and we calculate it using cross entropy loss. We want this loss to  decrease as we train our model.

## Backward Pass

The errors occured during forward pass are used to correct our weights and biases so that we make better predictions with the next steps.
We take the difference between predicted and actual labels as our error. We would then take derivative of our output activation and multiply it to our error which becomes our output layer gradient. We take the dot product of hidden layer's output and the output gradient and multiply it with learning rate. This becomes our gradient for output weights. We subtract this gradient from output weight which updates out output weights. For bias, we take the sum of output layer gradient and subtract it from our output bias. We repeat this process for hidden layers too.

## Loss function

I have used a cross entropy loss function. 

## Activation functions

1.) Sigmoid:

It is given by the formula: 1/(1+e^-x)

2.) Relu:

It is given by the formula: max(x,0)

3.) Tanh: 

It is given by the formula: 2 / (1 + e^(-2*x)) - 1. or (e^x - e^-x)/(e^x +e^-x)

4.) Identity:

It returns the input itself

## Initialization

1.) Xavier Initialization : https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

It returns a random uniform distribution from the range -1/sqrt(n) to 1/sqrt(n). We use xavier initialization mostly when we have a sigmoid or tanh activation function. We can also use it for identity function. For relu, xavier initialization runs into issues as mentioned in this article: https://www.machinecurve.com/index.php/2019/09/16/he-xavier-initialization-activation-functions-choose-wisely/#he-and-xavier-initialization-against-gradient-issues. So we use HE initialization instead.

2.) HE Initialization: https://medium.com/@shoray.goel/kaiming-he-initialization-a8d9ed0b5899

We initalise weights with a random uniform function and multiply them by sqrt(2/(n)), where n  = number of features.
