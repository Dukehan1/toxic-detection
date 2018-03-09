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

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 200
VECTOR_DIR = os.path.join('glove.twitter.27B.200d.txt')

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
        else:
            ws_to_idx[sp[0].decode('utf-8')] = k
            embeddings_matrix[k] = [float(x) for x in sp[1:]]
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

    ws_to_idx, embeddings_matrix = init_embedding()
    dict = {}
    dict['ws'] = ws_to_idx

    inputs_ret = []
    labels_ret = []
    mask_ret = []
    df = []
    df_idx = filter(lambda x: x != dev_id, range(1, 11))
    for i in df_idx:
        df.append(pd.read_csv(os.path.join('split', 'train-' + str(i) + '.csv')))
    df = pd.concat(df)
    term = 0
    for index, row in df.iterrows():
        sentence = normalize(row['comment_text'].decode('utf-8'))

        # mask
        length = len(sentence) if len(sentence) < MAX_SEQUENCE_LENGTH else MAX_SEQUENCE_LENGTH
        mask = [True] * length + [False] * (MAX_SEQUENCE_LENGTH - length)
        mask_ret.append(mask)

        # input
        x = {}
        x['ws'] = [0] * MAX_SEQUENCE_LENGTH
        for i, item in [('ws', sentence)]:
            # 超过最大长度则截断
            for k, v in enumerate(item[:MAX_SEQUENCE_LENGTH]):
                if dict[i].has_key(v):
                    x[i][k] = dict[i][v]
        x_flat = [x['ws']]
        inputs_ret.append(x_flat)

        labels_ret.append([
            row['toxic'],
            row['severe_toxic'],
            row['obscene'],
            row['threat'],
            row['insult'],
            row['identity_hate'],
        ])
        # term += 1
        # if term > 80: break
    training_set = [np.asarray(inputs_ret), np.asarray(mask_ret), np.asarray(labels_ret)]
    print "Finish loading training data"

    inputs_ret = []
    labels_ret = []
    mask_ret = []
    df = pd.read_csv(os.path.join('split', 'train-' + str(dev_id) + '.csv'))
    term = 0
    for index, row in df.iterrows():
        sentence = normalize(row['comment_text'].decode('utf-8'))

        # mask
        length = len(sentence) if len(sentence) < MAX_SEQUENCE_LENGTH else MAX_SEQUENCE_LENGTH
        mask = [True] * length + [False] * (MAX_SEQUENCE_LENGTH - length)
        mask_ret.append(mask)

        # input
        x = {}
        x['ws'] = [0] * MAX_SEQUENCE_LENGTH
        for i, item in [('ws', sentence)]:
            # 超过最大长度则截断
            for k, v in enumerate(item[:MAX_SEQUENCE_LENGTH]):
                if dict[i].has_key(v):
                    x[i][k] = dict[i][v]
        x_flat = [x['ws']]
        inputs_ret.append(x_flat)

        labels_ret.append([
            row['toxic'],
            row['severe_toxic'],
            row['obscene'],
            row['threat'],
            row['insult'],
            row['identity_hate'],
        ])
        # term += 1
        # if term > 200: break
    dev_set = [np.asarray(inputs_ret), np.asarray(mask_ret), np.asarray(labels_ret)]
    print "Finish loading dev data"

    config = Config()

    with tf.Graph().as_default():
        print "Building model...",
        start = time.time()
        model = LSTM_CNNModel(config, embeddings_matrix, os.path.join(model_dir, timestamp + ".model"))
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
            model.fit(session, saver, training_set, dev_set)
            print "took {:.2f} seconds\n".format(time.time() - start)
            print "Done!"

            inputs_ret = []
            mask_ret = []
            df = pd.read_csv(input_path_test)
            for index, row in df.iterrows():
                sentence = normalize(row['comment_text'].decode('utf-8'))

                # mask
                length = len(sentence) if len(sentence) < MAX_SEQUENCE_LENGTH else MAX_SEQUENCE_LENGTH
                mask = [True] * length + [False] * (MAX_SEQUENCE_LENGTH - length)
                mask_ret.append(mask)

                # input
                x = {}
                x['ws'] = [0] * MAX_SEQUENCE_LENGTH
                for i, item in [('ws', sentence)]:
                    # 超过最大长度则截断
                    for k, v in enumerate(item[:MAX_SEQUENCE_LENGTH]):
                        if dict[i].has_key(v):
                            x[i][k] = dict[i][v]
                x_flat = [x['ws']]
                inputs_ret.append(x_flat)
            test_set = [np.asarray(inputs_ret), np.asarray(mask_ret)]
            print "Finish loading test data"

            t = 0
            predict_raw = None
            # prevent OOM
            while t < len(test_set[0]):
                if predict_raw is None:
                    predict_raw = model.predict_on_batch(session, test_set[0][t:t + 1000],
                                                        test_set[1][t:t + 1000])
                else:
                    predict_raw = np.concatenate(
                        (predict_raw, model.predict_on_batch(session, test_set[0][t:t + 1000],
                                                            test_set[1][t:t + 1000])), axis=1)
                t += 1000
            (predict_raw, predict_proba) = predict_raw
            submission = pd.DataFrame.from_dict({'id': df['id']})
            class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
            for (id, class_name) in class_names.items():
                submission[class_name] = predict_proba[:, id]
            submission.to_csv(os.path.join(model_dir, 'submit.csv'), index=False)
            print "Finish test"

    return 0


class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs.
    """

    def add_placeholders(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, mask_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, mask_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, mask_batch)
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
    embed_size = EMBEDDING_DIM
    batch_size = 128
    n_epochs = 10
    lr = 0.01
    dropout = 0.5

    # open multitask
    label_num = 6

    """
    for NBT-CNN
    """
    filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_filters = 100

    """
    for LSTM
    """
    hidden_size = 200
    clip_gradients = True
    max_grad_norm = 5.

class LSTM_CNNModel(Model):
    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32, (None, 1, self.config.max_length))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.config.max_length))
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.label_num))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.mask_placeholder: mask_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):

        # word_embeddings = tf.convert_to_tensor(self.pretrained_word_embeddings)
        word_embeddings = tf.Variable(self.pretrained_word_embeddings)
        x_words = tf.nn.embedding_lookup(word_embeddings, self.inputs_placeholder[:, 0, :])

        x_raw = [x_words]
        x = tf.concat(x_raw, 2)

        preds = []
        for i in range(self.config.label_num):
            with tf.variable_scope("y-%s" % i):
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.hidden_size / 2,
                                                                                     initializer=tf.contrib.layers.xavier_initializer()),
                                                             output_keep_prob=self.dropout_placeholder)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.hidden_size / 2,
                                                                                     initializer=tf.contrib.layers.xavier_initializer()),
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
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, self.config.max_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID')
                        pooled_outputs.append(pooled)

                # Combine all the pooled features
                h_pool = tf.reduce_sum(pooled_outputs, 0)
                h_pool_flat = tf.reshape(h_pool, [-1, self.config.num_filters])
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_placeholder)

                U = tf.get_variable('U', (self.config.num_filters, 1),
                                    tf.float32, tf.contrib.layers.xavier_initializer())
                b2 = tf.get_variable('b2', (1,),
                                     tf.float32, tf.contrib.layers.xavier_initializer())

                pred = tf.matmul(h_drop, U) + b2
                preds.append(pred)
        preds = tf.concat(preds, 1)
        preds_proba = tf.nn.sigmoid(preds)
        return preds, preds_proba

    def add_loss_op(self, preds):
        '''
        loss = 0
        for i in range(self.config.label_num):
            loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[:, i], labels=self.labels_placeholder[:, i]))
        return loss / 6
        '''
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

    def train_on_batch(self, sess, inputs_batch, mask_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, mask_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

    def batch_evaluation(self, sess, examples):
        t = 0
        predict = None
        # prevent OOM
        while t < len(examples[0]):
            if predict is None:
                predict = self.predict_on_batch(sess, examples[0][t:t + 1000], examples[1][t:t + 1000])
            else:
                predict = np.concatenate((predict, self.predict_on_batch(sess, examples[0][t:t + 1000],
                                                                            examples[1][t:t + 1000])), axis=1)
            t += 1000
        (predict_raw, predict_proba) = predict
        auc = roc_auc_score(examples[2], predict_proba, average='macro')
        predict_tag = np.zeros(predict_raw.shape, dtype='int32')
        for i in range(len(predict_raw)):
            for j in range(len(predict_raw[i])):
                if predict_raw[i, j] >= 0:
                    predict_tag[i, j] = 1
        acc = accuracy_score(examples[2], predict_tag)
        f1 = f1_score(examples[2], predict_tag, average='weighted')
        print "- Acc: ", acc
        print "- F1: ", f1
        print "- AUC: ", auc
        return auc

    def run_epoch(self, sess, train_examples, dev_examples):
        for i, (inputs_batch, mask_batch, labels_batch) in enumerate(
                get_minibatches(train_examples, self.config.batch_size)):
            loss, grad_norm = self.train_on_batch(sess, inputs_batch, mask_batch, labels_batch)
            print "loss: ", loss, " grad_norm: ", grad_norm

        print "Evaluating on training set"
        self.batch_evaluation(sess, train_examples)
        print "Evaluating on dev set"
        dev_auc = self.batch_evaluation(sess, dev_examples)
        return dev_auc

    def fit(self, sess, saver, train_examples, dev_examples):
        best_dev_auc = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_auc = self.run_epoch(sess, train_examples, dev_examples)
            if dev_auc >= best_dev_auc:
                best_dev_auc = dev_auc
                if saver:
                    print '-' * 80
                    print "New best dev auc! Saving model in " + self.model_path
                    # 只存数据，不保存网络结构，否则model文件会非常大
                    saver.save(sess, self.model_path, write_meta_graph=False)
                    print '-' * 80
            print

    def __init__(self, config, pretrained_word_embeddings, model_path):
        self.pretrained_word_embeddings = pretrained_word_embeddings
        self.config = config
        self.model_path = model_path
        self.build()

if __name__ == "__main__":
    timestamp = get_timestamp()
    import sys
    model_dir = os.path.join(os.path.abspath('.'), sys.argv[1] + '_' + timestamp)
    mkdir_p(model_dir)
    error = open(os.path.join(model_dir, sys.argv[1] + '_error_' + timestamp + '.log'), 'w+')
    info = open(os.path.join(model_dir, sys.argv[1] + '_info_' + timestamp + '.log'), 'w+')
    sys.stderr = error
    sys.stdout = info

    test_set = os.path.join("test.csv")
    experiment(10, test_set, model_dir, timestamp)

    error.close()
    info.close()
