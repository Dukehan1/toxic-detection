# coding: utf-8

import os
import errno
from datetime import datetime
import numpy as np
import re
import pandas as pd
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Bidirectional, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, LSTM, Add
from keras.preprocessing import text, sequence
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import Callback

MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 300
MAX_FEATURES = 156853
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
    text = text.encode('utf-8')
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
    df_train = df_train[:80]
    X_train_raw = map(lambda t: normalize(t), df_train["comment_text"].fillna('').values)
    y_train = df_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading training data"

    df_dev = pd.read_csv(os.path.join('split', 'train-' + str(dev_id) + '.csv'))
    df_dev = df_dev[:200]
    X_dev_raw = map(lambda t: normalize(t), df_dev["comment_text"].fillna('').values)
    y_dev = df_dev[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    print "Finish loading dev data"

    df_test = pd.read_csv(os.path.join('test.csv'))
    df_test = df_test[:200]
    X_test_raw = map(lambda t: normalize(t), df_test["comment_text"].fillna('').values)
    print "Finish loading test data"

    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(X_train_raw) + list(X_dev_raw) + list(X_test_raw))
    X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=MAX_SEQUENCE_LENGTH)
    X_dev = sequence.pad_sequences(tokenizer.texts_to_sequences(X_dev_raw), maxlen=MAX_SEQUENCE_LENGTH)
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=MAX_SEQUENCE_LENGTH)

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
        pools = []
        for i in range(1, 11):
            conv = Conv1D(100, kernel_size=i, activation="relu", use_bias=True)(x)
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

    filepath = os.path.join(model_dir, "weights_base.best.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
    ra_val = RocAucEvaluation(validation_data=(X_dev, y_dev), interval=1)
    callbacks_list = [ra_val, checkpoint, early]

    model.fit(X_train, y_train, batch_size=128, epochs=4, validation_data=(X_dev, y_dev),
              callbacks=callbacks_list, verbose=2)
    # Loading model weights
    # model.load_weights(filepath)

    y_train_predict = model.predict(X_train, batch_size=1024, verbose=2)
    submission = pd.DataFrame.from_dict({'id': df_train['id']})
    submission['comment_text'] = X_train_raw
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = y_train_predict[:, id]
    submission.to_csv(os.path.join(model_dir, 'predict-keras-train.csv'), index=True)
    print "Finish train set prediction"

    y_dev_predict = model.predict(X_dev, batch_size=1024, verbose=2)
    submission = pd.DataFrame.from_dict({'id': df_dev['id']})
    submission['comment_text'] = X_dev_raw
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = y_dev_predict[:, id]
    submission.to_csv(os.path.join(model_dir, 'predict-keras-dev.csv'), index=True)
    print "Finish dev set prediction"

    y_test_predict = model.predict(X_test, batch_size=1024, verbose=2)
    submission = pd.DataFrame.from_dict({'id': df_test['id']})
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = y_test_predict[:, id]
    submission.to_csv(os.path.join(model_dir, 'submit.csv'), index=False)
    print "Finish test set prediction"

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
    import sys
    model_dir = os.path.join(os.path.abspath('.'), 'lstm_cnn_keras' + sys.argv[1])
    mkdir_p(model_dir)

    experiment(int(sys.argv[1]), model_dir)