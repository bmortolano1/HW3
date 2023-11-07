import numpy as np

class Perceptron:
    def __init__(self, size):
        self.weights = np.zeros(size)
        self.voted_perceptron_count = 0

    def set_weights(self, w):
        self.weights = w

    def get_weights(self):
        return self.weights

    def increment_count(self):
        self.voted_perceptron_count = self.voted_perceptron_count + 1

    def get_vp_count(self):
        return self.voted_perceptron_count

    def predict(self, x):
        return np.dot(self.weights, x)

    def update(self, update):
        self.weights = self.weights + update

    def predict_and_update(self, x, y, r):
        y_pred = self.predict(x)
        if y_pred*y <= 0:
            self.update(r*x*y)
        return y_pred