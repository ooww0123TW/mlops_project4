'''
diagnostics.py

Author: Wonseok Oh
Date: June 2023
'''
import os
import subprocess
import json
import pickle
import timeit

import pandas as pd

# Load config.json and get environment variables
with open('config.json','r', encoding='utf-8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

# Function to get model predictions
def model_predictions(data):
    '''
    model_predictions

    Input - data (pandas.dataFrame) - dataset
    Output - preds (numpy.ndarray) 
            : an array containing all predictions

    read the deployed model and a test dataset,
    calculate predictions
    '''

    state = data.drop(['corporation', 'exited'], axis=1)

    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(model_path, 'rb') as f_p:
        model = pickle.load(f_p)

    preds = model.predict(state)
    return preds

# Function to get summary statistics
def dataframe_summary():
    '''
    dataframe_summary

    Input - None
    Output - ret (list[pandas.core.series.Series]) 
            : a list containing all summary statistics

    calculate summary statistics here
    '''

    data_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    data = pd.read_csv(data_path)

    num_cols = data.select_dtypes(exclude='object').columns.tolist()
    means = data[num_cols].mean(axis=0)
    medians = data[num_cols].median(axis=0)
    stds = data[num_cols].std(axis=0)

    ret = [means, medians, stds]
    return ret

# Function to check missing data
def get_missing_data_ratio():
    '''
    get_missing_data_ratio
    Input: None
    Output: (pandas.core.series.Series)
            : a list containing na value ratio 
            with the same number of elements 
            as the number of columns in your dataset
    '''
    data_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    data = pd.read_csv(data_path)

    na_counts = data.isna().sum()

    return na_counts / data.shape[0]

# Function to get timings
def execution_time():
    '''
    execution_time

    Input: None
    Output: list[float]
        : a Python list consisting of two timing measurements in seconds: 
          one measurement for data ingestion, 
          and one measurement for model training in order.

    calculate timing of training.py and ingestion.py
    '''
    starttime1 = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_time = timeit.default_timer() - starttime1

    starttime2 = timeit.default_timer()
    os.system('python3 training.py')
    training_time = timeit.default_timer() - starttime2
    return [ingestion_time, training_time]

#Function to check dependencies
def outdated_packages_list():
    '''
    outdated_packages_list

    Input: None
    Output: ret (string) - a table with three columns: 
        the first column will show the name of a Python module that you're using;
        the second column will show the currently installed version of that Python module,
        and the third column will show the most recent available version of that Python module
    '''
    #get a list of outdated modeuls
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    outdated_str = outdated.decode('utf-8')
    lines = outdated_str.split('\n')

    header = lines[0]
    data = lines[1:]
    ret = pd.DataFrame([x.split() for x in data], columns = header.split())

    ret = ret.drop(columns="Type")
    return ret.to_string()


if __name__ == '__main__':
    test_data_full_path = os.path.join(test_data_path, 'testdata.csv')
    test_data = pd.read_csv(test_data_full_path)


    model_predictions(test_data)
    dataframe_summary()
    get_missing_data_ratio()
    execution_time()
    outdated_packages_list()
