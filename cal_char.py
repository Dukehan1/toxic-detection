# coding: utf-8

import os
import errno
from datetime import datetime

import re
from nltk.tokenize.treebank import TreebankWordTokenizer
import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode
from keras.preprocessing import text

MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 50
INFERENCE_BATCH_SIZE = 400

def get_timestamp():
    (dt, micro) = datetime.utcnow().strftime('%Y%m%d%H%M%S.%f').split('.')
    dt = "%s%03d" % (dt, int(micro) / 1000)
    return dt

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def normalize(text):
    text = text.decode('utf-8')
    text = re.sub(r'[a-zA-z]+://[^\s]*', '', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
    text = strip_accents_ascii(text)
    text = text.encode('utf-8')
    text = ' '.join(map(lambda x: x.lower(), TreebankWordTokenizer().tokenize(text)))
    return text

def experiment():
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="

    df = pd.read_csv(os.path.join('train.csv'))
    X_train_o = map(lambda x: normalize(x), df["comment_text"].fillna('').values)
    y_train = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading training data"

    df = pd.read_csv(os.path.join('test.csv'))
    X_test_o = map(lambda x: normalize(x), df["comment_text"].fillna('').values)
    print "Finish loading test data"

    tokenizer = text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(list(X_train_o) + list(X_test_o))
    X_train = tokenizer.texts_to_sequences(X_train_o)
    X_test = tokenizer.texts_to_sequences(X_test_o)

    print len(tokenizer.word_index)
    word_list = [''] * (len(tokenizer.word_index) + 1)
    for k, v in tokenizer.word_index.items():
        word_list[v] = (k, tokenizer.word_counts[k])
    i = 0
    for t in word_list[1:]:
        if tokenizer.word_counts[t[0]] > 1:
            i += 1
    print i
    print word_list[:i]
    print
    print
    print
    print word_list[i:]
    print len(X_train_o)
    print len(X_test_o)

    for k, v in enumerate(X_train):
        if len(v) > MAX_SEQUENCE_LENGTH:
            print k, ' ', y_train[k]
            # print v
            if list(y_train[k]) != [0, 0, 0, 0, 0, 0]:
                print X_train_o[k]
    print
    for k, v in enumerate(X_test):
        if len(v) > MAX_SEQUENCE_LENGTH:
            print k
            # print v
            # print X_test_o[k]

    return 0

if __name__ == "__main__":
    experiment()
