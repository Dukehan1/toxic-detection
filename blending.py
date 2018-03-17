# coding: utf-8

import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

INPUT_DIR = 'blending/'

predict_list = []
files = os.listdir(INPUT_DIR)
for f in files:
    if f.endswith(".csv"):
        predict_list.append(pd.read_csv(INPUT_DIR + f)[LABELS].values)
        print f

print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(6):
        predictions[:, i] = np.add(predictions[:, i], predict[:, i])
        # predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])
predictions /= len(predict_list)

submission = pd.read_csv(INPUT_DIR + files[1])
submission[LABELS] = predictions
submission.to_csv('blending.csv', index=False)