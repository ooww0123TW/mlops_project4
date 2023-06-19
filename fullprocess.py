'''
fullprocess.py

Author: Wonseok Oh
Date: June 2023
'''
import os
import json
import pickle

import pandas as pd
from sklearn import metrics

from ingestion import merge_multiple_dataframe
from training import train_model
from deployment import store_model_into_pickle
from diagnostics import model_predictions, dataframe_summary, get_missing_data_ratio, execution_time, outdated_packages_list
from reporting import score_model
import scoring

with open('config.json', encoding='utf-8', mode='r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])

# Check and read new data
# first, read ingestedfiles.txt
ingestedfiles_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
with open(ingestedfiles_path, "r", encoding='utf-8') as f_p:
    recorded_files = f_p.read().splitlines()

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files = []

for file in os.listdir(input_folder_path):
    if file not in recorded_files:
        new_files.append(file)

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) == 0:
    print("No any new file. Exit")
    exit()
print("New file detected. Proceed")

# Checking for model drift
score_path = os.path.join(prod_deployment_path, 'latestscore.txt')
with open(score_path, "r", encoding='utf-8') as f_p:
    score = float(f_p.read().strip())


# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
with open(model_path, "rb") as f_p:
    model = pickle.load(f_p)

merge_multiple_dataframe()
data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
y_test = data['exited']
x_test = data.drop(['corporation', 'exited'], axis=1)
y_preds = model.predict(x_test)

f1_score = metrics.f1_score(y_true=y_test, y_pred=y_preds)
print(f1_score)
is_drift=False
if f1_score <= score:
    is_drift = True

# Deciding whether to proceed, part 2

#if you found model drift, you should proceed. otherwise, do end the process here
if not is_drift:
    print("Drift non-detected. Exit")
    exit()
print("Drfit detected. Proceed")

# Re-training
train_model()
scoring.score_model()

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
store_model_into_pickle()

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
test_data_path = os.path.join(config['test_data_path'])
test_data_full_path = os.path.join(test_data_path, 'testdata.csv')
test_data = pd.read_csv(test_data_full_path)


model_predictions(test_data)
dataframe_summary()
get_missing_data_ratio()
execution_time()
outdated_packages_list()
score_model(data, model, 'confusionmatrix2.png')