import data_reader as dt
import perceptron_trainer as pt

if __name__ == '__main__':
    [train_features, train_labels] = dt.parse_file("./bank-note/train.csv")
    [test_features, test_labels] = dt.parse_file("./bank-note/test.csv")

    perc = pt.train(train_features, train_labels, 0.2, 10)
    pt.test(test_features, test_labels, perc)