from perceptron import Perceptron
import numpy as np

def train(features, labels, r, epochs):
    per = Perceptron(np.size(features, 1))
    rng = np.random.default_rng()
    indeces = np.arange(np.size(labels))

    for T in range(epochs):
        rng.shuffle(indeces)
        n_corr = 0
        n_incorr = 0
        for i in indeces:
            x = features[i]
            y = labels[i]
            y_pred = per.predict_and_update(x, y, r)

            if y_pred*y > 0:
                n_corr = n_corr+1
            else:
                n_incorr = n_incorr+1

    print("Final Accuracy: " + str(n_corr / (n_corr+n_incorr)))
    print("Weights: " + str(per.weights))

    return per

def test(features, labels, per):
    indeces = np.arange(np.size(labels))

    n_corr = 0
    n_incorr = 0
    for i in indeces:
        x = features[i]
        y = labels[i]
        y_pred = per.predict(x)

        if y_pred * y > 0:
            n_corr = n_corr + 1
        else:
            n_incorr = n_incorr + 1

    print("Final Accuracy: " + str(n_corr / (n_corr + n_incorr)))
    print("Weights: " + str(per.weights))