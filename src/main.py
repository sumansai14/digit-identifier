from library import Network
from mnsit_loader import load_data_wrapper

if __name__ == '__main__':
    training_data, validation_data, test_data = load_data_wrapper()
    model = Network([784, 30, 10])
    model.train(training_data, 200, 200, 3.0, test_data=test_data)
