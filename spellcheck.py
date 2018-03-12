# coding: utf-8

import os
import errno
from datetime import datetime
from nltk.tokenize.treebank import TreebankWordTokenizer
import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii
from textblob import TextBlob

MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 300
MAX_FEATURES = 150000
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
    text = str(TextBlob(text).correct())
    return text

def experiment():
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="

    dfs_idx = range(1, 11)
    for i in dfs_idx:
        df = pd.read_csv(os.path.join('split', 'train-' + str(i) + '.csv'))
        for index, row in df.iterrows():
            row["comment_text"] = normalize(row["comment_text"])
        df.to_csv(os.path.join('split', 'sc-train-' + str(i) + '.csv'))

    df = pd.read_csv(os.path.join('train.csv'))
    for index, row in df.iterrows():
        row["comment_text"] = normalize(row["comment_text"])
    df.to_csv(os.path.join('sc-train.csv'), index=False)

    df = pd.read_csv(os.path.join('test.csv'))
    for index, row in df.iterrows():
        row["comment_text"] = normalize(row["comment_text"])
    df.to_csv(os.path.join('sc-test.csv'), index=False)

    return 0

if __name__ == "__main__":
    experiment()
