'''
scoring.py

Author: Wonseok Oh
Date: June 2023
'''
import pickle
import os
import json

import pandas as pd

from sklearn import metrics


# Load config.json and get path variables
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

# Function for model scoring


def score_model():
    '''
    score_model
    Input: None
    Output: f1_score (numpy.float64)

    take a trained model, load test data, and calculate an F1 score 
    for the model relative to the test data.
    write the result to the latestscore.txt file
    '''
    data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as f_p:
        model = pickle.load(f_p)

    y_test = data['exited']
    x_test = data.drop(['corporation', 'exited'], axis=1)
    y_preds = model.predict(x_test)

    f1_score = metrics.f1_score(y_true=y_test, y_pred=y_preds)

    with open(os.path.join(model_path, 'latestscore.txt'), 'w', encoding='utf-8') as f_p:
        f_p.write(str(f1_score))

    return f1_score

if __name__ == '__main__':
    score_model()
