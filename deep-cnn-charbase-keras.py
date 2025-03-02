# coding: utf-8

import os
import errno
from datetime import datetime
import re
import pandas as pd
import numpy as np
from keras.backend import one_hot
from nltk import TreebankWordTokenizer
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv1D, BatchNormalization, Activation, \
    MaxPooling1D, Dropout, Flatten
from keras.preprocessing import text, sequence
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import Callback

MAX_SEQUENCE_LENGTH = 2000
MAX_FEATURES = 68

INFERENCE_BATCH_SIZE = 128

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

def experiment(dev_id, model_dir):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="

    df_train = []
    df_train_idx = filter(lambda x: x != dev_id, range(1, 11))
    for i in df_train_idx:
        df_train.append(pd.read_csv(os.path.join('split', 'train-' + str(i) + '.csv')))
    df_train = pd.concat(df_train)
    # df_train = df_train[:80]
    X_train_raw = map(lambda t: normalize(t), df_train["comment_text"].fillna('').values)
    y_train = df_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading training data"

    df_dev = pd.read_csv(os.path.join('split', 'train-' + str(dev_id) + '.csv'))
    # df_dev = df_dev[:200]
    X_dev_raw = map(lambda t: normalize(t), df_dev["comment_text"].fillna('').values)
    y_dev = df_dev[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading dev data"

    df_test = pd.read_csv(os.path.join('test.csv'))
    # df_test = df_test[:200]
    X_test_raw = map(lambda t: normalize(t), df_test["comment_text"].fillna('').values)
    print "Finish loading test data"

    tokenizer = text.Tokenizer(num_words=MAX_FEATURES, char_level=True)
    tokenizer.fit_on_texts(list(X_train_raw) + list(X_dev_raw) + list(X_test_raw))
    X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=MAX_SEQUENCE_LENGTH)
    X_dev = sequence.pad_sequences(tokenizer.texts_to_sequences(X_dev_raw), maxlen=MAX_SEQUENCE_LENGTH)
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=MAX_SEQUENCE_LENGTH)

    word_index = tokenizer.word_index
    print word_index
    valid_features = min(MAX_FEATURES, len(word_index)) + 1
    print valid_features
    embeddings_matrix = np.r_[np.zeros((1, valid_features - 1)), np.eye(valid_features - 1, dtype=int)]
    print embeddings_matrix

    def get_model():
        inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
        x = Embedding(valid_features, valid_features - 1, weights=[embeddings_matrix], trainable=False)(inp)
        for i in range(1, 3):
            x = Conv1D(filters=256, kernel_size=7, use_bias=True)(x)
            x = BatchNormalization()(x)
            x = Activation(activation="relu")(x)
            x = MaxPooling1D(pool_size=3)(x)
        for i in range(1, 4):
            x = Conv1D(filters=256, kernel_size=3, use_bias=True)(x)
            x = BatchNormalization()(x)
            x = Activation(activation="relu")(x)
        x = Conv1D(filters=256, kernel_size=3, use_bias=True)(x)
        x = BatchNormalization()(x)
        x = Activation(activation="relu")(x)
        x = MaxPooling1D(pool_size=3)(x)
        x = Flatten()(x)
        for i in range(1, 3):
            x = Dense(1024, activation="relu")(x)
            x = Dropout(0.5)(x)
        outp = Dense(6, activation="sigmoid")(x)
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

    model.fit(X_train, y_train, batch_size=128, epochs=24, validation_data=(X_dev, y_dev),
              callbacks=callbacks_list, verbose=2)
    # 注意：要加载保存的最优模型
    model.load_weights(filepath)

    y_train_predict = model.predict(X_train, batch_size=INFERENCE_BATCH_SIZE, verbose=2)
    submission = pd.DataFrame.from_dict({'id': df_train['id']})
    submission['comment_text'] = X_train_raw
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = y_train_predict[:, id]
    submission.to_csv(os.path.join(model_dir, 'predict-keras-train.csv'), index=True)
    print "- AUC: ", roc_auc_score(y_train, y_train_predict)
    print "Finish train set prediction"

    y_dev_predict = model.predict(X_dev, batch_size=INFERENCE_BATCH_SIZE, verbose=2)
    submission = pd.DataFrame.from_dict({'id': df_dev['id']})
    submission['comment_text'] = X_dev_raw
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = y_dev_predict[:, id]
    submission.to_csv(os.path.join(model_dir, 'predict-keras-dev.csv'), index=True)
    print "- AUC: ", roc_auc_score(y_dev, y_dev_predict)
    print "Finish dev set prediction"

    y_test_predict = model.predict(X_test, batch_size=INFERENCE_BATCH_SIZE, verbose=2)
    submission = pd.DataFrame.from_dict({'id': df_test['id']})
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = y_test_predict[:, id]
    submission.to_csv(os.path.join(model_dir, 'submit.csv'), index=False)
    print "Finish test set prediction"

    return 0

class RocAucMetricCallback(Callback):
    def __init__(self, predict_batch_size=INFERENCE_BATCH_SIZE, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size=predict_batch_size
        self.include_on_batch=include_on_batch

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
    model_dir = os.path.join(os.path.abspath('.'), 'char_deep_cnn_keras_' + sys.argv[1])
    mkdir_p(model_dir)

    experiment(int(sys.argv[1]), model_dir)
