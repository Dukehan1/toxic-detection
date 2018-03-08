# coding: utf-8

import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
import nltk
import pandas as pd

def experiment_svm_dev_set(input_path_training):
    df = pd.read_csv(input_path_training)
    X = []
    y = []
    tokenizer = nltk.TreebankWordTokenizer()
    for index, row in df.iterrows():
        X.append(' '.join(tokenizer.tokenize(row['comment_text'].decode('utf-8'))))
        y.append(row['toxic'])
    y = np.asarray(y)
    print "Finish loading data"

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    X_train, X_test, y_train, y_test = None, None, None, None
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break
    print "Finish splitting data"

if __name__ == "__main__":
    training_set = os.path.join("train.csv")
    experiment_svm_dev_set(training_set)