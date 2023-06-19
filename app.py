'''
app.py

Author: Wonseok Oh
Date: June 2023
'''

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import model_predictions, dataframe_summary, get_missing_data_ratio, execution_time, outdated_packages_list
from scoring import score_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    '''
    predict
    Input: None
    Output: (str) - prediction of the input

    call the prediction function you created in Step 3
    '''
    data_path = request.args.get('data')
    data = pd.read_csv(data_path)
    preds = model_predictions(data)
    return str(preds) + '\n'

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    '''
    score
    Input: None
    Output: (str) - f1_score

    check the score of the deployed model
    add return value (a single F1 score number)
    '''
    f1 = score_model()

    return str(f1) + '\n'

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():     
    '''
    stats
    Input: None
    Output: (str) - means, medians, stds information of the dataset

    check means, medians, and modes for each column
    '''   
    summary = dataframe_summary()
    return str(summary) + '\n'

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnosis():
    '''
    diagnosis
    Input: None
    Output: ret (str) - timing, missing data, dependency check output

    check timing and percent NA values
    '''
    timing = execution_time()
    missing = get_missing_data_ratio()
    outdated = outdated_packages_list()
    ret = str(timing) + '\n' + str(missing) + '\n' + outdated
    return ret

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
