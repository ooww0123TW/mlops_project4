'''
reporting.py

Author: Wonseok Oh
Date: June, 2023
'''

import os
import json
import pickle

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Load config.json and get path variables
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_csv_path = os.path.join(config['test_data_path'])
dep_model_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path'])

# Function for reporting


def score_model(data, model, filename):
    '''
    score_model

    Input: None
    Output: None

    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    '''

    y_test = data['exited']
    x_test = data.drop(['corporation', 'exited'], axis=1)
    y_preds = model.predict(x_test)

    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_preds)

    plt.imshow(conf_mat, interpolation='nearest', cmap='viridis')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    fig_path = os.path.join(output_model_path, filename)
    plt.savefig(fig_path)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(test_csv_path, 'testdata.csv'))
    with open(os.path.join(dep_model_path, 'trainedmodel.pkl'), 'rb') as f_p:
        trainedmodel = pickle.load(f_p)

    score_model(df, trainedmodel, 'confusionmatrix.png')
