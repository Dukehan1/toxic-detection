# coding: utf-8

import os
import pandas as pd
import numpy as np

def experiment(path):
    df = None
    result = []
    for p in path:
        df = pd.read_csv(os.path.join(p))
        result.append(df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values)
        print "Finish loading data ", p
    result = np.sum(result, axis=0, dtype=np.float32)
    result = np.divide(result, len(path))

    submission = pd.DataFrame.from_dict({'id': df['id']})
    class_names = {0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}
    for (id, class_name) in class_names.items():
        submission[class_name] = result[:, id]
    submission.to_csv(os.path.join('voting.csv'), index=False)
    print "Finish voting"

if __name__ == "__main__":
    path = [
        'ml/lr_submit.csv',
        'submit.csv'
    ]
    experiment(path)