import random
import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    """
    The network object takes in the layer diemensions and initalizes the weights.
    The imputs are passed in the train method
    """
    def __init__(self, layer_dims):
        self.num_layers = len(layer_dims)
        self.dimensions = layer_dims
        self.biases = [np.zeros((y, 1)) for y in layer_dims[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_dims[:-1], layer_dims[1:])]

    def labelize(self, data):
        """Tranform a matrix (where each column is a data) into an list that contains the argmax of each item."""
        return np.argmax(data, axis=0)

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """
        Takes the inputs and in each epoch - shuffles them, splits them into mini batches
        does the forward prop, calculates the gradients, and updates the weights.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        # Change the shape of the test data
        test_x, test_y = map(list, zip(*test_data))
        test_x = np.reshape(test_x, (n_test, self.dimensions[0])).T
        test_y = np.reshape(test_y, (n_test, self.dimensions[-1])).T
        test_labels = self.labelize(test_y)
        train_x, train_y = map(list, zip(*training_data))
        train_x = np.reshape(train_x, (n, self.dimensions[0])).T
        train_y = np.reshape(train_y, (n, self.dimensions[-1])).T
        train_labels = self.labelize(train_y)
        # Start the training
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                x, y = map(list, zip(*mini_batch))
                x = np.reshape(x, (mini_batch_size, self.dimensions[0])).T
                y = np.reshape(y, (mini_batch_size, self.dimensions[-1])).T
                # Forward Propagation
                final_activation, activation_cache, linear_cache = self.forward_propagation(x, learning_rate)

                # Backward Prop
                gradients = self.backward_propagation(final_activation, y, activation_cache, linear_cache)

                # Update weights
                self.update_parameters(gradients, learning_rate, mini_batch_size)

                # Print the training error and the test error
            if j % 10 == 0 and j > 0:
                print "Error percentage after %s iterations" % j
                print "The error on the training set is {0:.2f}%".format(self.evaluate(
                    self.labelize(self.forward_propagation(train_x, learning_rate)[0]), train_labels) * 100)
                print "The error on the test set is {0:.2f}%".format(self.evaluate(
                    self.labelize(self.forward_propagation(test_x, learning_rate)[0]), test_labels
                ) * 100)

    def evaluate(self, a, y):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        return (np.sum(a != y) / (len(a) + 0.0))

    def forward_propagation(self, x, learning_rate):
        """
        Take the input through forward propagation and cache the linear output
        and sigmoid output to use it in backward prop
        """
        activation_cache = [x]
        linear_cache = []
        a = x
        for w, b in zip(self.weights, self.biases):
            # print a.shape, w.shape, b.shape
            z = np.dot(w, a) + b
            linear_cache.append(z)
            a = sigmoid(z)
            activation_cache.append(a)
        return (a, activation_cache, linear_cache)

    def backward_propagation(self, final_activation, y, activation_cache, linear_cache):
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        dal = self.cost_derivative(activation_cache[-1], y)
        for l in range(1, self.num_layers):
            dzl = self.sigmoid_backward(dal, linear_cache[-l])
            dw[-l] = np.dot(dzl, activation_cache[-l - 1].T)
            db[-l] = np.sum(dzl, axis=1, keepdims=True)
            dal = np.dot(self.weights[-l].transpose(), dzl)
        return (dw, db)

    def sigmoid_backward(self, da, cache):
        z = cache
        dz = np.multiply(da, sigmoid_prime(z))
        return dz

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def update_parameters(self, gradients, learning_rate, mini_batch_size):
        dw, db = gradients
        self.weights = [w - (learning_rate / mini_batch_size) * dw_l for w, dw_l in zip(self.weights, dw)]
        self.biases = [b - (learning_rate / mini_batch_size) * db_l for b, db_l in zip(self.biases, db)]

