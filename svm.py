# coding: utf-8

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
import sklearn.externals.joblib as jl
import nltk
import pandas as pd

VECTOR_DIR = os.path.join('w2v_w10_min5_c0_v200.txt')

EMBEDDING_DIM = 200

def experiment_svm(input_path_training):
    df = pd.read_csv(input_path_training)
    X = []
    y = []
    tokenizer = nltk.TreebankWordTokenizer()
    for index, row in df.iterrows():
        X.append(' '.join(tokenizer.tokenize(row['comment_text'].decode('utf-8'))))
        y.append([
            row['toxic'],
            row['severe_toxic'],
            row['obscene'],
            row['threat'],
            row['insult'],
            row['identity_hate']
        ])
    y = np.asarray(y)
    print "Finish loading data"
    '''
    tv = TfidfVectorizer(stop_words='english', min_df=0.00001)
    tv.fit_transform(X)
    print len(tv.get_feature_names())
    '''

    classifier = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english', min_df=0.001)),
        ('clf', OneVsRestClassifier(SVC(cache_size=5000, class_weight='balanced', kernel="linear", C=10, verbose=2))),
    ])
    classifier.fit(X, y)
    print "Finish training"

    predict = classifier.predict(X)
    acc = accuracy_score(y, predict)
    f1 = f1_score(y, predict, average='weighted')
    print "Accuracy For Training Set: " + str(acc)
    print "F1 For Training Set: " + str(f1)
    jl.dump(classifier, os.path.join("svm"))

def experiment_svm_cv(input_path_training):
    df = pd.read_csv(input_path_training)
    X = []
    y = []
    tokenizer = nltk.TreebankWordTokenizer()
    for index, row in df.iterrows():
        X.append(' '.join(tokenizer.tokenize(row['comment_text'].decode('utf-8'))))
        y.append([
            row['toxic'],
            row['severe_toxic'],
            row['obscene'],
            row['threat'],
            row['insult'],
            row['identity_hate']
        ])
    y = np.asarray(y)
    print "Finish loading data"

    param_grid = [
        {
            'clf__estimator__kernel': ['linear'],
            'clf__estimator__C': [1, 10, 100],
            'vect__min_df': [1e-3, 1e-4, 1e-5]
        },
        {
            'clf__estimator__kernel': ['rbf'],
            'clf__estimator__gamma': [1e-1, 1],
            'clf__estimator__C': [10, 100],
            'vect__min_df': [1e-3, 1e-4, 1e-5]
        }
    ]
    classifier = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english')),
        ('clf', OneVsRestClassifier(SVC(cache_size=5000, class_weight='balanced'))),
    ])
    gs = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1, verbose=2)
    gs.fit(X, y)
    print "Finish training"

    predict = gs.predict(X)
    acc = accuracy_score(y, predict)
    f1 = f1_score(y, predict, average='weighted')
    print gs.cv_results_
    print "The best parameters are %s with a score of %0.2f" % (gs.best_params_, gs.best_score_)
    print "Accuracy For Training Set: " + str(acc)
    print "F1 For Training Set: " + str(f1)
    jl.dump(gs, os.path.join("svm_cv"))

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
    experiment_svm(training_set)