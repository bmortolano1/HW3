import data_reader as dt
import perceptron_trainer as pt

if __name__ == '__main__':
    [train_features, train_labels] = dt.parse_file("./bank-note/train.csv")
    [test_features, test_labels] = dt.parse_file("./bank-note/test.csv")

    # Standard Perceptron
    perc = pt.train_std_perceptron(train_features, train_labels, 1, 10)
    pt.test_std_perceptron(test_features, test_labels, perc)

    # Voted Perceptron
    percs = pt.train_voted_perceptron(train_features, train_labels, 1, 10)
    pt.test_voted_perceptron(train_features, train_labels,percs)

    # Standard Perceptron
    per_avg = pt.train_avg_perceptron(train_features, train_labels, 1, 10)
    pt.test_avg_perceptron(test_features, test_labels, per_avg)