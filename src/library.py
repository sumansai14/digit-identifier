import numpy as np
import random


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, layer_dims):
        """
        This is the class for the network object.

        The way network object is created is, it takes the hyperparameters and constructs the corresponding network
        Let's start with the following hyperparameters
        :layer_dims - diemensions of the no. of hidden layers
        :learning_rate

        """
        self.num_layers = len(layer_dims)
        self.dimensions = layer_dims
        self.biases = [np.zeros((y, 1)) for y in layer_dims[1:]]
        self.weights = [np.random.randn(y, x) * 0.01 for x, y in zip(layer_dims[:-1], layer_dims[1:])]

    def predict(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None, cost_function='logistic_regression', sgd=False):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                x, y = map(list, zip(*mini_batch))
                # It is important to match the diemensions in while vectorizing the data.
                x = np.reshape(x, (self.dimensions[0], mini_batch_size))
                y = np.reshape(y, (self.dimensions[-1], mini_batch_size))
                if sgd is True:
                    final_activation = self.update_mini_batch(mini_batch, learning_rate, cost_function)
                    final_activation = np.reshape(final_activation, y.shape)
                else:
                    final_activation, activation_cache, linear_cache = self.forward_propagation(x, learning_rate)
                    gradients = self.backward_propagation(
                        final_activation,
                        y, activation_cache, linear_cache, cost_function=cost_function)
                    self.update_parameters(gradients, learning_rate, mini_batch_size)
            if j % 10 == 0:
                cost = self.compute_cost(final_activation, y, cost_function)  # Used for gradient checking
                print "Cost after epoch %i: %f" % (j, cost)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)

    def update_mini_batch(self, mini_batch, learning_rate, cost_function):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        activation_cache = []
        for x, y in mini_batch:
            dwi, dbi, final_activation = self.sgd(x, y, cost_function)
            db = [old_db + ndb for old_db, ndb in zip(db, dbi)]
            dw = [old_dw + ndw for old_dw, ndw in zip(dw, dwi)]
            activation_cache.append(final_activation)
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, dw)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, db)]
        return activation_cache

    def sgd(self, x, y, cost_function):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activation_cache = [x]
        linear_cache = []
        # Forward prop
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            linear_cache.append(z)
            activation = sigmoid(z)
            activation_cache.append(activation)
        # Backward Prop
        if cost_function == 'logistic_regression':
            dal = -(np.divide(y, activation_cache[-1]) - np.divide(1 - y, 1 - activation_cache[-1]))
        elif cost_function == 'mse':
            dal = self.cost_derivative(activation_cache[-1], y)
        for l in range(1, self.num_layers):
            dzl = self.sigmoid_backward(dal, linear_cache[-l])
            dw[-l] = np.dot(dzl, activation_cache[-l - 1].transpose())
            db[-l] = dzl
            dal = np.dot(self.weights[-l].transpose(), dzl)
        return (dw, db, activation_cache[-1])

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.predict(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def compute_cost(self, a, y, cost_function):
        if cost_function == 'logistic_regression':
            cost = -np.mean(y * np.log(a) + np.log(1 - y) * np.log(1 - a))
        elif cost_function == 'mse':
            cost = -np.mean(np.power(a - y, 2)) / 2
        return cost

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def forward_propagation(self, x, learning_rate):
        activation_cache = [x]
        a = x
        linear_cache = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            linear_cache.append(z)
            a = sigmoid(z)
            activation_cache.append(a)
        return (a, activation_cache, linear_cache)

    def sigmoid_backward(self, da, cache):
        z = cache
        dz = da * sigmoid_prime(z)
        return dz

    def backward_propagation(self, final_activation, y, activation_cache, linear_cache, cost_function):
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        if cost_function == 'logistic_regression':
            dal = -(np.divide(y, final_activation) - np.divide(1 - y, 1 - final_activation))
        elif cost_function == 'mse':
            dal = self.cost_derivative(activation_cache[-1], y)
        for l in range(1, self.num_layers):
            dzl = self.sigmoid_backward(dal, linear_cache[-l])
            dw[-l] = np.dot(dzl, activation_cache[-l - 1].transpose())
            db[-l] = np.sum(dzl, axis=1, keepdims=True)
            dal = np.dot(self.weights[-l].transpose(), dzl)
        return (dw, db)

    def update_parameters(self, gradients, learning_rate, mini_batch_size):
        dw, db = gradients
        self.weights = [w - (learning_rate / mini_batch_size) * dw_l for w, dw_l in zip(self.weights, dw)]
        self.biases = [b - (learning_rate / mini_batch_size) * db_l for b, db_l in zip(self.biases, db)]
