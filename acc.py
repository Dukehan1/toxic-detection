import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df_true = pd.read_csv('result/train-10.csv')
    df_predict = pd.read_csv('result/dev-predict.csv')
    y_true = df_true[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    y_predict = df_predict[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    accuracy_idx = np.mean(np.equal(y_true, np.round(y_predict)), -1)
    for k, v in enumerate(accuracy_idx):
        if v < 1:
            print df_predict.ix[k]['id']
            print df_predict.ix[k]['comment_text']
            print y_predict[k]
            print y_true[k]

