# coding: utf-8

import os
from sklearn.model_selection import KFold
import pandas as pd

def experiment(input_path_training):
    df = pd.read_csv(input_path_training)
    X = []
    y = []
    for index, row in df.iterrows():
        X.append(row['comment_text'])
        y.append([
            row['toxic'],
            row['severe_toxic'],
            row['obscene'],
            row['threat'],
            row['insult'],
            row['identity_hate'],
        ])
    print "Finish loading data"

    skf = KFold(n_splits=10, random_state=2333, shuffle=True)
    i = 1
    for train_index, dev_index in skf.split(X, y):
        df_dev = df.ix[dev_index]
        df_dev.to_csv('train-' + str(i) + '.csv')
        i += 1
    print "Finish splitting data"

if __name__ == "__main__":
    training_set = os.path.join("train.csv")
    experiment(training_set)