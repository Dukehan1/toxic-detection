# coding: utf-8

import os
from sklearn.feature_extraction.text import TfidfVectorizer, strip_accents_ascii
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
from nltk.tokenize.treebank import TreebankWordTokenizer
import pandas as pd
import re

def normalize(text):
    text = text.decode('utf-8')
    text = re.sub(r'[a-zA-z]+://[^\s]*', '', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
    text = ' '.join(map(lambda x: x.lower(), TreebankWordTokenizer().tokenize(text)))
    return text

def experiment(model_path):
    train = pd.read_csv('../train.csv').fillna('')
    test = pd.read_csv('../test.csv').fillna('')

    train_text = train['comment_text']
    test_text = test['comment_text']
    all_text = pd.concat([train_text, test_text])
    all_text = map(lambda t: normalize(t), all_text)
    print 1

    word_vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        min_df=2,
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
        ngram_range=(1, 2))
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    print 2

    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=50000)
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)
    print 3

    train_features = hstack([train_char_features, train_word_features])
    test_features = hstack([test_char_features, test_word_features])

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    submission = pd.DataFrame.from_dict({'id': test['id']})
    for class_name in class_names:
        train_target = train[class_name]
        classifier = LogisticRegression(C=0.1, max_iter=200, solver='lbfgs', class_weight='balanced', n_jobs=-1, verbose=2)

        classifier.fit(train_features, train_target)
        auc = roc_auc_score(train_target, classifier.predict_proba(train_features)[:, 1])
        print "AUC For Training Set: " + str(auc)

        submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    submission.to_csv(model_path + '_submit.csv', index=False)
    print "Finish test"

if __name__ == "__main__":
    experiment('lr')