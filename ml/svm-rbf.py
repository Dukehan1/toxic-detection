# coding: utf-8

import os
from sklearn.feature_extraction.text import TfidfVectorizer, strip_accents_ascii
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import sklearn.externals.joblib as jl
from nltk.tokenize.treebank import TreebankWordTokenizer
import pandas as pd
from sklearn.svm import SVC

'''
param_grid = [
    {
        'clf__estimator__kernel': ['linear'],
        'clf__estimator__C': [1e-1, 1, 10, 100]
    },
    {
        'clf__estimator__kernel': ['rbf'],
        'clf__estimator__C': [1e-1, 1, 10, 100],
        'clf__estimator__gamma': [1e-1, 1]
    }
]
'''

def normalize(text):
    text = strip_accents_ascii(text)
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
    predict_proba = classifier.predict_proba(X)
    auc = roc_auc_score(y, predict_proba, average='macro')
    print "AUC For Training Set: " + str(auc)
    jl.dump(classifier, os.path.join(model_path))
    print "Finish saving model"

    df = pd.read_csv(input_path_test)
    X_test = []
    for index, row in df.iterrows():
        X_test.append(normalize(row['comment_text'].decode('utf-8')))
    print "Finish loading test data"

    predict_proba = classifier.predict_proba(X_test)
    submission = pd.DataFrame.from_dict({'id': df['id']})
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = predict_proba[:, id]
    submission.to_csv(model_path + '_submit.csv', index=False)
    print "Finish test"

if __name__ == "__main__":
    training_set = os.path.join("../train.csv")
    test_set = os.path.join("../test.csv")
    experiment(training_set, test_set, 'svm-rbf', OneVsRestClassifier(SVC(cache_size=60000, class_weight='balanced',
                                                                      probability=True, kernel='rbf', C=10, gamma=1, verbose=2)))