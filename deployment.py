'''
deployment.py

Author: Wonseok Oh
Date: June 2023
'''
import os
import json
import shutil


# Load config.json and correct path variable
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

# function for deployment


def store_model_into_pickle():
    '''
    store_model_into_pickle

    Input: None
    Output: None

    copy the latest pickle file, the latestscore.txt value,
    and the ingestfiles.txt file into the deployment directory
    '''
    shutil.copy2(
        os.path.join(
            dataset_csv_path,
            'ingestedfiles.txt'),
        prod_deployment_path)
    shutil.copy2(
        os.path.join(
            model_path,
            'trainedmodel.pkl'),
        prod_deployment_path)
    shutil.copy2(
        os.path.join(
            model_path,
            'latestscore.txt'),
        prod_deployment_path)


if __name__ == '__main__':
    store_model_into_pickle()
