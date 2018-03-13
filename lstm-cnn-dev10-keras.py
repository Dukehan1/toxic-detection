# coding: utf-8

import os
import errno
from datetime import datetime
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer
import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Bidirectional, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, LSTM
from keras.preprocessing import text, sequence
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import Callback

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
    # text = str(TextBlob(text).correct())
    return text

def experiment(dev_id, model_dir, timestamp):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="

    df = []
    df_idx = filter(lambda x: x != dev_id, range(1, 11))
    for i in df_idx:
        df.append(pd.read_csv(os.path.join('split', 'train-' + str(i) + '.csv')))
    df = pd.concat(df)
    # df = df[:80]
    X_train = map(lambda x: normalize(x), df["comment_text"].fillna('').values)
    y_train = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading training data"

    '''
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, train_size=0.9, random_state=233)
    print "Finish loading dev data"
    '''
    df = pd.read_csv(os.path.join('split', 'train-' + str(dev_id) + '.csv'))
    # df = df[:200]
    X_dev = map(lambda x: normalize(x), df["comment_text"].fillna('').values)
    y_dev = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading dev data"

    submission = pd.read_csv(os.path.join('test.csv'))
    # df = df[:200]
    X_test = map(lambda x: normalize(x), submission["comment_text"].fillna('').values)
    print "Finish loading test data"

    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(X_train) + list(X_dev) + list(X_test))
    X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_SEQUENCE_LENGTH)
    X_dev = sequence.pad_sequences(tokenizer.texts_to_sequences(X_dev), maxlen=MAX_SEQUENCE_LENGTH)
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_SEQUENCE_LENGTH)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_dict = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(VECTOR_DIR))

    word_index = tokenizer.word_index
    valid_features = min(MAX_FEATURES, len(word_index) + 1)
    embeddings_matrix = np.random.uniform(-1, 1, (valid_features, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None: embeddings_matrix[i] = embedding_vector

    def get_model():
        inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
        x = Embedding(valid_features, EMBEDDING_DIM, weights=[embeddings_matrix])(inp)
        x = SpatialDropout1D(0.5)(x)
        x = Bidirectional(LSTM(200, return_sequences=True, recurrent_dropout=0.5))(x)
        x = Conv1D(100, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        outp = Dense(6, activation="sigmoid")(conc)
        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=1e-3),
                      metrics=['accuracy'])

        return model

    model = get_model()

    filepath = os.path.join(model_dir, "weights_base.best.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
    ra_val = RocAucEvaluation(validation_data=(X_dev, y_dev), interval=1)
    callbacks_list = [ra_val, checkpoint, early]

    model.fit(X_train, y_train, batch_size=128, epochs=4, validation_data=(X_dev, y_dev),
              callbacks=callbacks_list, verbose=2)
    # Loading model weights
    model.load_weights(filepath)
    print('Predicting....')
    y_test = model.predict(X_test, batch_size=1024, verbose=2)

    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
    submission.to_csv(os.path.join(model_dir, 'submit.csv'), index=False)
    print "Finish test"

    return 0

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

if __name__ == "__main__":
    '''
    timestamp = '20180311135230454'
    import sys
    model_dir = os.path.join('dev10-d10-b128_20180311135230454')
    mkdir_p(model_dir)
    '''
    timestamp = get_timestamp()
    import sys
    model_dir = os.path.join(os.path.abspath('.'), sys.argv[1] + '_' + timestamp)
    mkdir_p(model_dir)

    experiment(10, model_dir, timestamp)
