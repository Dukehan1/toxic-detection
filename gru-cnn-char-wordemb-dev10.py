# coding: utf-8

import os
import errno
from datetime import datetime
import numpy as np
import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.metrics import roc_auc_score
from nltk.tokenize.treebank import TreebankWordTokenizer
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Bidirectional, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, LSTM, Add, BatchNormalization, Activation
from keras.layers.core import Reshape
from keras.preprocessing import text, sequence
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import Callback

MAX_SEQUENCE_LENGTH = 300
MAX_WORD_LENGTH = 20
EMBEDDING_DIM = 300
CHAR_EMBEDDING_DIM = 50
MAX_FEATURES = 156853
CHAR_HIDDEN_SIZE = 100
WORD_HIDDEN_SIZE = 200
DROP_OUT = 0.5
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
    text = text.decode('utf-8')
    text = re.sub(r'[a-zA-z]+://[^\s]*', '', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
    text = strip_accents_ascii(text)
    text = text.encode('utf-8')
    return text

class MappingDict(object):

    def __init__(self, min_count = 0):
        self.min_count = 0
        self.c = Counter()

    def update(self, iter):
        self.c.update(iter)

    def freeze(self):
        self.d = dict(self.c)
        self.l = [x for x in list(self.c) if self.d[x] >= self.min_count]
        self.m = dict(zip(self.l, range(1, len(self.l) + 1)))

    def conv(self, string_or_int):
        if isinstance(string_or_int, int):
            if string_or_int < len(self.l):
                return self.l[string_or_int]
            else:
                raise IndexError('out of index')
        elif isinstance(string_or_int, list):
            return [self.conv(i) for i in string_or_int]
        else:
            if not self.d.has_key(string_or_int) or self.d[string_or_int] < self.min_count:
                return 0
            else:
                return self.m[string_or_int]

    def __len__(self):
        return len(self.l)



def experiment(dev_id, model_dir):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="

    df_train = []
    df_train_idx = filter(lambda x: x != dev_id, range(1, 11))
    for i in df_train_idx:
        df_train.append(pd.read_csv(os.path.join('splitraw', 'train-' + str(i) + '.csv')))
    df_train = pd.concat(df_train)
    # df_train = df_train[:80]
    X_train = map(lambda t: normalize(t), df_train["comment_text"].fillna('').values)
    y_train = df_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading training data"

    df_dev = pd.read_csv(os.path.join('splitraw', 'train-' + str(dev_id) + '.csv'))
    # df_dev = df_dev[:200]
    X_dev = map(lambda t: normalize(t), df_dev["comment_text"].fillna('').values)
    y_dev = df_dev[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading dev data"

    df_test = pd.read_csv(os.path.join('test.csv'))
    # df_test = df_test[:200]
    X_test = map(lambda t: normalize(t), df_test["comment_text"].fillna('').values)
    print "Finish loading test data"

    ## tokenize
    word_map = MappingDict(min_count=1)
    char_map = MappingDict()
    for train_data in (X_train, X_dev, X_test):
        for data_i in xrange(0, len(train_data)):
            train_data[data_i] = TreebankWordTokenizer().tokenize(train_data[data_i])
            word_map.update(train_data[data_i])
            char_map.update(''.join(train_data[data_i]))

    word_map.freeze()
    char_map.freeze()

    for train_data in (X_train, X_dev, X_test):
        for data_i in xrange(0, len(train_data)):
            sentence = train_data[data_i]
            feature = {"word": [], "char": []}
            for word_i in xrange(0, MAX_SEQUENCE_LENGTH):
                word = train_data[data_i][word_i] if word_i < len(sentence) else ''
                feature["word"].append(word_map.conv(word) if word_i < len(sentence) else 0)
                charactors = list(word)
                for c_i in xrange(0, MAX_WORD_LENGTH):
                    feature["char"].append(char_map.conv(charactors[c_i]) if c_i < len(charactors) else 0)
            train_data[data_i] = feature

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_dict = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(VECTOR_DIR))

    valid_features = min(MAX_FEATURES, len(word_map) + 1)
    embeddings_matrix = np.random.uniform(-1, 1, (valid_features, EMBEDDING_DIM))
    for word in word_map.l:
        i = word_map.conv(word)
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None: embeddings_matrix[i] = embedding_vector

    def get_model():
        inp = Input(shape=(MAX_SEQUENCE_LENGTH,), name='word_input')
        input_chars = Input(shape=(MAX_SEQUENCE_LENGTH * MAX_WORD_LENGTH,), name='char_input')
        input_chars = Reshape((MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH))(input_chars)
        c = Embedding(len(char_map), CHAR_EMBEDDING_DIM, embeddings_initializer='uniform',
                      input_length=MAX_SEQUENCE_LENGTH * MAX_WORD_LENGTH)(input_chars)
        c = SpatialDropout1D(DROP_OUT)(c)
        c = Reshape((MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH))(c)
        c = Bidirectional(LSTM(CHAR_HIDDEN_SIZE, return_sequences=True, recurrent_dropout=DROP_OUT))(c)

        x = Embedding(valid_features, EMBEDDING_DIM, weights=[embeddings_matrix])(inp)
        x = concatenate([c, x], axis=-1)
        x = SpatialDropout1D(0.5)(x)
        pools = []
        for i in range(1, 11):
            conv = Conv1D(100, kernel_size=i, use_bias=True)(x)
            conv = BatchNormalization()(conv)
            conv = Activation(activation="relu")(conv)
            conv = Conv1D(100, kernel_size=i, use_bias=True)(conv)
            conv = BatchNormalization()(conv)
            conv = Activation(activation="relu")(conv)
            avg_pool = GlobalAveragePooling1D()(conv)
            max_pool = GlobalMaxPooling1D()(conv)
            pool = concatenate([avg_pool, max_pool])
            pools.append(pool)
        outp = Dense(6, activation="sigmoid")(Add()(pools))
        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=1e-3),
                      metrics=['accuracy'])

        return model

    model = get_model()

    ra_val = RocAucMetricCallback()  # include it before EarlyStopping!
    filepath = os.path.join(model_dir, "weights_base.best.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='roc_auc_val', verbose=2, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="roc_auc_val", mode="max", patience=5)
    callbacks_list = [ra_val, checkpoint, early]

    model.fit({"word_input" : X_train["word"], "char_input": X_train["char"]}, y_train, batch_size=128, epochs=8, validation_data=({"word_input" : X_dev["word"], "char_input": X_dev["char"]}, y_dev),
              callbacks=callbacks_list, verbose=2)
    # 注意：要加载保存的最优模型
    model.load_weights(filepath)

    y_train_predict = model.predict({"word_input" : X_train["word"], "char_input": X_train["char"]}, batch_size=1024, verbose=2)
    submission = pd.DataFrame.from_dict({'id': df_train['id']})
    submission['comment_text'] = X_train
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = y_train_predict[:, id]
    submission.to_csv(os.path.join(model_dir, 'predict-keras-train.csv'), index=True)
    print "- AUC: ", roc_auc_score(y_train, y_train_predict)
    print "Finish train set prediction"

    y_dev_predict = model.predict(X_dev, batch_size=1024, verbose=2)
    submission = pd.DataFrame.from_dict({'id': df_dev['id']})
    submission['comment_text'] = X_dev
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = y_dev_predict[:, id]
    submission.to_csv(os.path.join(model_dir, 'predict-keras-dev.csv'), index=True)
    print "- AUC: ", roc_auc_score(y_dev, y_dev_predict)
    print "Finish dev set prediction"

    y_test_predict = model.predict(X_test, batch_size=1024, verbose=2)
    submission = pd.DataFrame.from_dict({'id': df_test['id']})
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = y_test_predict[:, id]
    submission.to_csv(os.path.join(model_dir, 'submit.csv'), index=False)
    print "Finish test set prediction"

    return 0


class RocAucMetricCallback(Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if self.include_on_batch:
            logs['roc_auc_val'] = float('-inf')
            if self.validation_data:
                logs['roc_auc_val'] = roc_auc_score(self.validation_data[1],
                                                    self.model.predict(self.validation_data[0],
                                                                       batch_size=self.predict_batch_size))

    def on_train_begin(self, logs={}):
        if not 'roc_auc_val' in self.params['metrics']:
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val'] = float('-inf')
        if self.validation_data:
            logs['roc_auc_val'] = roc_auc_score(self.validation_data[1],
                                                self.model.predict(self.validation_data[0],
                                                                   batch_size=self.predict_batch_size))
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch + 1, logs['roc_auc_val']))


if __name__ == "__main__":
    import sys

    model_dir = os.path.join(os.path.abspath('.'), 'char-word_keras_' + sys.argv[1])
    mkdir_p(model_dir)

    experiment(int(sys.argv[1]), model_dir)
