'''
training.py

Author: Wonseok Oh
Date: June 2023
'''

import pickle
import os
import json

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression


# Load config.json and get path variables
with open('config.json', encoding='utf-8', mode='r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


# Function for training the model
def train_model():
    '''
    train_model
    Input: None
    Output: None

    Read data in finaldata.csv and train a logistic regression model.
    Save the trained model
    '''
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    target = data['exited']
    state = data.drop(['corporation', 'exited'], axis=1)

    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # fit the logistic regression to your data
    model.fit(state, target)

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as f_p:
        pickle.dump(model, f_p)


if __name__ == '__main__':
    train_model()
