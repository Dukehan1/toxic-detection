# coding: utf-8

import os
import errno
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import re

from nltk import TreebankWordTokenizer
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from keras.preprocessing import text, sequence

MAX_SEQUENCE_LENGTH = 1000
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

    config = Config()

    with tf.Graph().as_default():
        print "Building model...",
        start = time.time()
        model = LSTM_CNNModel(config, embeddings_matrix, os.path.join(model_dir, "weight.model"))
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        # If you are using an old version of TensorFlow, you may have to use
        # this initializer instead.
        # init = tf.initialize_all_variables()
        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
            # writer = tf.summary.FileWriter("logs/", session.graph)
            session.run(init)

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            start = time.time()
            model.fit(session, saver, [X_train, y_train], [X_dev, y_dev])
            print "took {:.2f} seconds\n".format(time.time() - start)
            print "Done!"

            # 注意：要加载保存的最优模型
            saver.restore(session, os.path.join(model_dir, "weight.model"))

            train_set = [X_train]
            t = 0
            predict_raw = None
            # prevent OOM
            while t < len(train_set[0]):
                if predict_raw is None:
                    predict_raw = model.predict_on_batch(session, train_set[0][t:t + INFERENCE_BATCH_SIZE])
                else:
                    predict_raw = np.concatenate(
                        (predict_raw, model.predict_on_batch(session, train_set[0][t:t + INFERENCE_BATCH_SIZE])), axis=1)
                t += INFERENCE_BATCH_SIZE
            (predict_raw, predict_proba) = predict_raw
            submission = pd.DataFrame.from_dict({'id': df_train['id']})
            submission['comment_text'] = X_train_raw
            class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
            for (id, class_name) in class_names.items():
                submission[class_name] = predict_proba[:, id]
            submission.to_csv(os.path.join(model_dir, 'predict-tf-train.csv'))
            print "- AUC: ", roc_auc_score(y_train, predict_proba)
            print "Finish train set prediction"

            dev_set = [X_dev]
            t = 0
            predict_raw = None
            # prevent OOM
            while t < len(dev_set[0]):
                if predict_raw is None:
                    predict_raw = model.predict_on_batch(session, dev_set[0][t:t + INFERENCE_BATCH_SIZE])
                else:
                    predict_raw = np.concatenate(
                        (predict_raw, model.predict_on_batch(session, dev_set[0][t:t + INFERENCE_BATCH_SIZE])), axis=1)
                t += INFERENCE_BATCH_SIZE
            (predict_raw, predict_proba) = predict_raw
            submission = pd.DataFrame.from_dict({'id': df_dev['id']})
            submission['comment_text'] = X_dev_raw
            class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
            for (id, class_name) in class_names.items():
                submission[class_name] = predict_proba[:, id]
            submission.to_csv(os.path.join(model_dir, 'predict-tf-dev.csv'))
            print "- AUC: ", roc_auc_score(y_dev, predict_proba)
            print "Finish dev set prediction"

            test_set = [X_test]
            t = 0
            predict_raw = None
            # prevent OOM
            while t < len(test_set[0]):
                if predict_raw is None:
                    predict_raw = model.predict_on_batch(session, test_set[0][t:t + INFERENCE_BATCH_SIZE])
                else:
                    predict_raw = np.concatenate(
                        (predict_raw, model.predict_on_batch(session, test_set[0][t:t + INFERENCE_BATCH_SIZE])), axis=1)
                t += INFERENCE_BATCH_SIZE
            (predict_raw, predict_proba) = predict_raw
            submission = pd.DataFrame.from_dict({'id': df_test['id']})
            class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
            for (id, class_name) in class_names.items():
                submission[class_name] = predict_proba[:, id]
            submission.to_csv(os.path.join(model_dir, 'submit.csv'), index=False)
            print "Finish test set prediction"

    return 0


class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs.
    """

    def add_placeholders(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        pred, pred_proba = sess.run([self.pred, self.pred_proba], feed_dict=feed)
        return pred, pred_proba

    def build(self):
        self.add_placeholders()
        self.pred, self.pred_proba = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    max_length = MAX_SEQUENCE_LENGTH
    batch_size = 128
    n_epochs = 4
    lr = 0.001
    dropout = 0.8

    # open multitask
    label_num = 6

    """
    for CNN
    """
    filter_sizes = [3, 4, 5, 6, 7, 8, 9]
    num_filters = 100

    """
    for LSTM
    """
    hidden_size = 400
    clip_gradients = True
    max_grad_norm = 5.

class LSTM_CNNModel(Model):
    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32, (None, self.config.max_length))
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.label_num))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):

        word_embeddings = tf.Variable(self.pretrained_word_embeddings, dtype=tf.float32, trainable=False)
        x = tf.nn.embedding_lookup(word_embeddings, self.inputs_placeholder)
        x = tf.nn.dropout(x, self.dropout_placeholder)

        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.config.hidden_size / 2),
                                                     output_keep_prob=self.dropout_placeholder)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.config.hidden_size / 2),
                                                     output_keep_prob=self.dropout_placeholder)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        h_lstm = tf.expand_dims(tf.concat(outputs, 2), -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.hidden_size, 1, self.config.num_filters]

                W = tf.get_variable('W', filter_shape,
                                    tf.float32, tf.contrib.layers.xavier_initializer())
                b1 = tf.get_variable('b1', (self.config.num_filters,),
                                     tf.float32, tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(
                    h_lstm,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b1))
                # Max-pooling over the outputs
                pooled_max = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                pooled_avg = tf.nn.avg_pool(
                    h,
                    ksize=[1, self.config.max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                pooled_outputs.append(tf.concat([pooled_avg, pooled_max], 3))

        # Combine all the pooled features
        h_pool = tf.reduce_sum(pooled_outputs, 0)
        h_pool_flat = tf.reshape(h_pool, [-1, self.config.num_filters * 2])

        preds = []
        for i in range(self.config.label_num):
            with tf.variable_scope("y-%s" % i):

                U = tf.get_variable('U', (self.config.num_filters * 2, 1),
                                    tf.float32, tf.contrib.layers.xavier_initializer())
                b2 = tf.get_variable('b2', (1,),
                                     tf.float32, tf.contrib.layers.xavier_initializer())

                pred = tf.matmul(h_pool_flat, U) + b2
                preds.append(pred)
        preds = tf.concat(preds, 1)
        preds_proba = tf.nn.sigmoid(preds)
        return preds, preds_proba

    def add_loss_op(self, preds):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=self.labels_placeholder))

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)

        grad_var = optimizer.compute_gradients(loss)
        grad = [i[0] for i in grad_var]
        var = [i[1] for i in grad_var]

        self.grad_norm = tf.global_norm(grad)
        if self.config.clip_gradients:
            grad, self.grad_norm = tf.clip_by_global_norm(grad, self.config.max_grad_norm)

        train_op = optimizer.apply_gradients(zip(grad, var))
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

    def batch_evaluation(self, sess, examples):
        t = 0
        predict = None
        # prevent OOM
        while t < len(examples[0]):
            if predict is None:
                predict = self.predict_on_batch(sess, examples[0][t:t + INFERENCE_BATCH_SIZE])
            else:
                predict = np.concatenate((predict, self.predict_on_batch(sess, examples[0][t:t + INFERENCE_BATCH_SIZE])), axis=1)
            t += INFERENCE_BATCH_SIZE
        (predict_raw, predict_proba) = predict
        auc = roc_auc_score(examples[1], predict_proba, average='macro')
        y_predict = np.round(predict_proba)
        acc = np.mean(np.equal(examples[1], y_predict))
        f1 = f1_score(examples[1], y_predict, average='weighted')
        print "- Acc: ", acc
        print "- F1: ", f1
        print "- AUC: ", auc
        return auc

    def evaluation(self, sess, saver, train_examples, dev_examples):
        print "Evaluating on training set"
        self.batch_evaluation(sess, train_examples)
        print "Evaluating on dev set"
        dev_auc = self.batch_evaluation(sess, dev_examples)

        if dev_auc > self.best_dev_auc:
            self.best_dev_auc = dev_auc
            if saver:
                print '-' * 80
                print "New best dev auc! Saving model in " + self.model_path
                saver.save(sess, self.model_path)
                print '-' * 80
        print

    def fit(self, sess, saver, train_examples, dev_examples):
        for epoch in range(self.config.n_epochs):
            epoch_loss = 0
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            for i, (inputs_batch, labels_batch) in enumerate(
                    get_minibatches(train_examples, self.config.batch_size)):
                if i % 200 == 0:
                    self.evaluation(sess, saver, train_examples, dev_examples)
                loss, grad_norm = self.train_on_batch(sess, inputs_batch, labels_batch)
                epoch_loss += loss
                print "loss: ", loss, " grad_norm: ", grad_norm
            self.evaluation(sess, saver, train_examples, dev_examples)
            print "epoch_loss: ", epoch_loss

    def __init__(self, config, pretrained_word_embeddings, model_path):
        self.pretrained_word_embeddings = pretrained_word_embeddings
        self.config = config
        self.model_path = model_path
        self.best_dev_auc = 0
        self.build()

if __name__ == "__main__":
    import sys
    model_dir = os.path.join(os.path.abspath('.'), 'char_lstm_cnn_tf_' + sys.argv[1])
    mkdir_p(model_dir)

    experiment(int(sys.argv[1]), model_dir)
