# coding: utf-8

import os
from sklearn.feature_extraction.text import TfidfVectorizer, strip_accents_ascii
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import sklearn.externals.joblib as jl
from nltk.tokenize.treebank import TreebankWordTokenizer
import pandas as pd
import re

def normalize(text):
    text = re.sub(r'[a-zA-z]+://[^\s]*', '', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
    text = ' '.join(map(lambda x: x.lower(), TreebankWordTokenizer().tokenize(text)))
    return text

def experiment(input_path_training, input_path_test, model_path, clf):
    df = pd.read_csv(input_path_training)
    X = []
    y = []
    for index, row in df.iterrows():
        X.append(normalize(row['comment_text'].decode('utf-8')))
        y.append([
            row['toxic'],
            row['severe_toxic'],
            row['obscene'],
            row['threat'],
            row['insult'],
            row['identity_hate']
        ])
    y = np.asarray(y)
    print "Finish loading training data"

    classifier = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)),
        ('clf', clf),
    ])
    classifier.fit(X, y)
    print "Finish training"

    predict = classifier.predict(X)
    acc = accuracy_score(y, predict)
    f1 = f1_score(y, predict, average='weighted')
    print "Accuracy For Training Set: " + str(acc)
    print "F1 For Training Set: " + str(f1)
    predict = classifier.predict(X)
    auc = roc_auc_score(y, predict, average='macro')
    print "AUC For Training Set: " + str(auc)
    jl.dump(classifier, os.path.join(model_path))
    print "Finish saving model"

    df = pd.read_csv(input_path_test)
    X_test = []
    for index, row in df.iterrows():
        X_test.append(normalize(row['comment_text'].decode('utf-8')))
    print "Finish loading test data"

    predict = classifier.predict(X_test)
    submission = pd.DataFrame.from_dict({'id': df['id']})
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = predict[:, id]
    submission.to_csv(model_path + '_submit.csv', index=False)
    print "Finish test"

if __name__ == "__main__":
    training_set = os.path.join("../train.csv")
    test_set = os.path.join("../test.csv")
    experiment(training_set, test_set, 'svm', OneVsRestClassifier(SGDClassifier(n_iter=20, class_weight='balanced',
                                                                               n_jobs=-1, verbose=2)))