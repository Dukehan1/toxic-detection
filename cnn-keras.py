# coding: utf-8

import os
import errno
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer
import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.optimizers import Adam

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 50
VECTOR_DIR = os.path.join('en-cw.txt')

INFERENCE_BATCH_SIZE = 400

def init_embedding():
    # build embedding dict
    print "Loading pretrained embeddings...",
    start = time.time()
    ws_to_idx = {}
    embeddings_matrix = None
    for k, line in enumerate(open(VECTOR_DIR).readlines()):
        sp = line.strip().split()
        if k == 0:
            embeddings_matrix = np.zeros((int(sp[0]) + 1, EMBEDDING_DIM), dtype='float32')
            embeddings_matrix[0] = np.random.uniform(-1, 1, EMBEDDING_DIM)
        else:
            ws_to_idx[sp[0].decode('utf-8')] = k
            embeddings_matrix[k] = np.asarray([float(x) for x in sp[1:]])
    print "took {:.2f} seconds\n".format(time.time() - start)
    return ws_to_idx, embeddings_matrix

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
    text = strip_accents_ascii(text)
    text = map(lambda x: x.lower(), TreebankWordTokenizer().tokenize(text))
    return text

def experiment(dev_id, input_path_test, model_dir, timestamp):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="

    df = []
    df_idx = filter(lambda x: x != dev_id, range(1, 11))
    for i in df_idx:
        df.append(pd.read_csv(os.path.join('split', 'train-' + str(i) + '.csv')))
    df = pd.concat(df)
    df = df[:80]
    X_train = df["comment_text"].fillna("fillna").values
    y_train = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading training data"

    df = pd.read_csv(os.path.join('split', 'train-' + str(dev_id) + '.csv'))
    df = df[:200]
    X_dev = df["comment_text"].fillna("fillna").values
    y_dev = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading dev data"

    max_features = 30000

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_dev))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_dev = tokenizer.texts_to_sequences(X_dev)
    x_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    x_dev = sequence.pad_sequences(X_dev, maxlen=MAX_SEQUENCE_LENGTH)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_dict = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(VECTOR_DIR))

    word_index = tokenizer.word_index
    valid_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((valid_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    def get_model():
        inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
        x = Embedding(valid_words, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(GRU(80, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        outp = Dense(6, activation="sigmoid")(conc)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=1e-2),
                      metrics=['accuracy'])

        return model

    model = get_model()
    hist = model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_dev, y_dev), verbose=2)

    return 0

if __name__ == "__main__":
    timestamp = get_timestamp()
    import sys
    model_dir = os.path.join(os.path.abspath('.'), sys.argv[1] + '_' + timestamp)
    mkdir_p(model_dir)

    test_set = os.path.join("test.csv")
    experiment(10, test_set, model_dir, timestamp)
