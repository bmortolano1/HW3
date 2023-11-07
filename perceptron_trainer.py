from perceptron import Perceptron
import numpy as np

def train_std_perceptron(features, labels, r, epochs):
    print("Enter Standard Perceptron Training Function.")

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
    print("Weights: " + str(per.weights) + str("\n"))

    return per

def test_std_perceptron(features, labels, per):
    print("Enter Standard Perceptron Testing Function.")

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
    print("Weights: " + str(per.weights) + str("\n"))

def get_perceptron_votes(x, percs):
    running_sum = 0

    for perc in percs:
        running_sum = running_sum + perc.get_vp_count() * np.sign(perc.predict(x))

    return np.sign(running_sum)

def train_voted_perceptron(features, labels, r, epochs):
    print("Enter Voted Perceptron Training Function.")

    pers = np.array([Perceptron(np.size(features, 1))])
    rng = np.random.default_rng()
    indeces = np.arange(np.size(labels))

    for T in range(epochs):
        rng.shuffle(indeces)
        for i in indeces:
            x = features[i]
            y = labels[i]
            y_pred = pers[-1].predict(x)

            if y_pred * y <= 0:
                pers = np.append(pers, Perceptron(np.size(features, 1)))
                pers[-1].set_weights(pers[-2].get_weights())
                pers[-1].predict_and_update(x, y, r)
            else:
                pers[-1].increment_count()

    print(str([x.weights for x in pers]))
    print(str([x.get_vp_count() for x in pers]))

    n_corr = 0
    n_incorr = 0

    for i in range(np.size(labels)):
        x = features[i]
        y = labels[i]
        y_pred = get_perceptron_votes(x, pers)

        if y_pred * y <= 0:
            n_incorr = n_incorr + 1
        else:
            n_corr = n_corr + 1

    print("Final Accuracy: " + str(n_corr / (n_corr + n_incorr)) + "\n")

    return pers

def test_voted_perceptron(features, labels, pers):
    print("Enter Voted Perceptron Testing Function.")

    n_corr = 0
    n_incorr = 0

    for i in range(np.size(labels)):
        x = features[i]
        y = labels[i]
        y_pred = get_perceptron_votes(x, pers)

        if y_pred * y <= 0:
            n_incorr = n_incorr + 1
        else:
            n_corr = n_corr + 1

    print("Final Accuracy: " + str(n_corr / (n_corr + n_incorr)) + "\n")

    return pers

def train_avg_perceptron(features, labels, r, epochs):
    print("Enter Average Perceptron Training Function.")

    per = Perceptron(np.size(features, 1))
    rng = np.random.default_rng()
    indeces = np.arange(np.size(labels))

    a = np.zeros(np.size(features,1))
    k = 0

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

            a = a + per.get_weights()
            k = k+1

    per.set_weights(a/k)

    print("Final Accuracy: " + str(n_corr / (n_corr+n_incorr)))
    print("Weights: " + str(per.weights) + str("\n"))

    return per

def test_avg_perceptron(features, labels, per):
    print("Enter Average Perceptron Testing Function.")

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
    print("Weights: " + str(per.weights) + str("\n"))