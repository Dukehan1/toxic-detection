# coding: utf-8

import os
import errno
from datetime import datetime
from nltk.tokenize.treebank import TreebankWordTokenizer
import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii
from keras.preprocessing import text
from textblob import TextBlob

MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 300
MAX_FEATURES = 140285
VECTOR_DIR = os.path.join('glove.840B.300d.txt')

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
    text = strip_accents_ascii(text.decode('utf-8'))
    text = text.encode('utf-8')
    text = ' '.join(map(lambda x: x.lower(), TreebankWordTokenizer().tokenize(text)))
    # text = str(TextBlob(text).correct())
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

    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
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
    print word_list[:MAX_FEATURES]
    print
    print
    print
    print word_list[MAX_FEATURES:]
    print len(X_train_o)
    print len(X_test_o)

    for k, v in enumerate(X_train):
        if len(v) > 500:
            print k, ' ', y_train[k]
            # print v
            if list(y_train[k]) != [0, 0, 0, 0, 0, 0]:
                print X_train_o[k]
    print
    for k, v in enumerate(X_test):
        if len(v) > 500:
            print k
            # print v
            # print X_test_o[k]

    return 0

if __name__ == "__main__":
    experiment()
